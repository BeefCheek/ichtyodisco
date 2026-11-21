# ichtyodisco

Un aquarium qui s'observe lui-même, en direct : tracking des poissons, **profondeur
monoculaire** (Apple Depth Pro / Depth Anything V2), analyse comportementale qui **apprend
la « normale » de l'aquarium heure par heure**, dashboard web, **livestream RTMP**
(YouTube / Twitch) et sortie **OSC** pour piloter un synthé.

```
caméra ──► YOLO + ByteTrack ──► profondeur (async) ──► moteur d'analyse ──► overlay vidéo ──► RTMP (YouTube/Twitch)
                                                          │                        └──────► dashboard web (MJPEG + SSE)
                                                          ├─► SQLite (séries, événements, modèle appris)
                                                          └─► OSC (SuperCollider, Max, Pd, TouchDesigner)
```

## Démarrage rapide

```bash
uv sync --extra depth                 # dépendances (+ transformers pour la profondeur)
uv run ichtyodisco --demo             # aquarium simulé : ni caméra ni GPU nécessaires
# → ouvrir http://localhost:8765
```

Avec la vraie caméra :

```bash
uv run ichtyodisco -s 0 --model models/fish.pt --depth depth-anything-v2-small
uv run ichtyodisco -s 0 --model models/fish.pt --depth depth-pro --device cuda:0
uv run ichtyodisco -s enregistrement.mp4           # rejouer une vidéo
uv run ichtyodisco -c config/local.toml            # tout via un fichier (voir config/example.toml)
```

## Le livestream

Deux façons, combinables :

1. **Direct depuis ichtyodisco** (le plus simple, tourne sans écran) :
   ```bash
   uv run ichtyodisco -s 0 --model models/fish.pt --stream "rtmp://a.rtmp.youtube.com/live2/<clé>"
   ```
   ffmpeg reçoit la vidéo annotée (overlay compris) à cadence constante — même si l'inférence
   ralentit, le flux reste fluide — avec une piste audio silencieuse (YouTube l'exige) et
   redémarre tout seul si le réseau coupe. Sur GPU NVIDIA, mettre `encoder = "h264_nvenc"`.
   Ne commitez jamais la clé de stream : mettez-la dans `config/local.toml` (ignoré par git)
   ou passez-la en argument.
2. **Via OBS** (pour une mise en scène plus libre) : source *Navigateur* sur
   `http://localhost:8765/video.mjpg` (ou la caméra brute), plus une seconde source
   *Navigateur* sur `http://localhost:8765/overlay` (widget transparent : humeur, activité vs
   normale, dernier événement), puis stream depuis OBS.

Le **dashboard** (`http://localhost:8765`) montre la vidéo annotée, les KPI avec leur écart à
la normale (σ), la courbe d'activité avec la bande « normale apprise », le **rythme appris sur
24 h**, les états de nage, une vue de dessus en direct (gauche-droite × profondeur), les
cartes d'occupation, le journal d'événements et les **profils des résidents** — qu'on peut
renommer d'un clic (idéal pour laisser le chat baptiser les poissons). Par défaut il n'écoute
que sur `127.0.0.1` : il n'a pas d'authentification, ne l'exposez pas tel quel sur Internet.

## La profondeur

Une seule caméra ne voit pas la profondeur ; un modèle **monoculaire** l'estime :

| modèle (`--depth`) | vitesse | licence | quand l'utiliser |
|---|---|---|---|
| `depth-anything-v2-small` | rapide (temps réel sur GPU modeste) | Apache-2.0 | **défaut**, OK pour un stream public |
| `depth-pro` (Apple) | ~0,3 s/image sur gros GPU, plusieurs s sur CPU | Apple ASCL, **recherche uniquement** | qualité max, usage perso |
| `depth-anything-v2-base/large` | moyen / lent | CC-BY-NC-4.0 | non commercial |

> Note : il n'existe pas (à ce jour) de « Depth Pro 2 » publié par Apple — le modèle actuel est
> Depth Pro (ICLR 2025). « V2 » désigne Depth Anything V2. N'importe quel checkpoint
> Hugging Face `depth-estimation` fonctionne via `--depth <repo/modèle>`.

Comment c'est intégré :
- **Asynchrone** : le tracking tourne à chaque image ; la profondeur tourne sur un thread à
  part (4 Hz max par défaut) sur la dernière image. Les poissons changent lentement de
  profondeur, chaque image réutilise la carte la plus récente.
- **Calibrée sur l'aquarium** : la profondeur monoculaire n'est connue qu'à une échelle près
  (et la vitre réfracte). On normalise donc entre un plan « avant » et un plan « fond »
  (percentiles robustes de la carte, lissés dans le temps) → `z ∈ [0 vitre, 1 fond]`.
- **Par poisson** : percentile bas du centre de la boîte (le poisson est l'objet le plus
  proche dans sa boîte), puis lissage EWMA pour que le bruit de profondeur ne se transforme
  pas en fausse vitesse.
- Les vitesses sont en **3D** et en **longueurs de corps par seconde** (BL/s), l'unité qui a un
  sens biologique ; `tank_depth_ratio` (profondeur / largeur de la cuve) met z à l'échelle.

## Un modèle qui évolue avec les habitants

Rien n'est codé en dur sur « ce qui est normal » : tout est **appris en continu** et
**oublie progressivement** (demi-vie configurable, 72 h par défaut), donc le modèle suit les
changements de saison, de nouveaux arrivants, un nouvel horaire de repas…

- **Baseline circadienne** : pour chaque métrique (activité, cohésion, polarisation, hauteur,
  profondeur, nombre de poissons…), une moyenne/écart-type **par heure de la journée**. Les
  alertes sont des écarts en σ à la normale *de cette heure-là* : 3 BL/s est normal à l'heure
  du repas, alarmant à 3 h du matin. Le header du dashboard affiche la progression
  (« modèle appris 14/24 h »).
- **États de nage adaptatifs** : repos / croisière / exploration / sprint, avec des seuils qui
  sont des quantiles de la distribution des vitesses *de cet aquarium* (un betta et un banc
  de danios n'ont pas la même définition du sprint), plus une hystérésis anti-clignotement.
- **Résidents** : les IDs du tracker changent à chaque occlusion et à chaque redémarrage. Une
  ré-identification (signature couleur du corps + continuité spatio-temporelle) les recolle en
  identités stables, avec des profils long terme : temps visible, vitesse moyenne et record,
  profondeur et hauteur préférées, zone favorite, part du temps au repos.
- **Social** : cohésion du banc, distance au plus proche voisin, polarisation (nage
  synchronisée), paires proches.
- **Occupation** : cartes de chaleur vue de face et vue de dessus, avec décroissance.
- **Persistance** : tout (baselines, seuils, résidents, cartes) est sauvegardé dans SQLite
  toutes les minutes et rechargé au démarrage — l'aquarium reprend là où il en était.

### Événements détectés

| événement | règle (tenue N secondes + cooldown) |
|---|---|
| `feeding` / `feeding_end` | pic d'activité (> 2σ) + groupe en surface |
| `agitation` / `lethargy` | activité > 3σ pendant 20 s / < −2,5σ pendant 90 s |
| `tight_shoal` | banc anormalement resserré (cohésion < −2,5σ) — réaction d'alerte ? |
| `schooling` | polarisation > 0,85 : nage coordonnée |
| `surface_crowding` | > 60 % des poissons en surface 2 min hors repas → vérifier O₂ / température |
| `chase` | deux poissons proches, rapides, même cap ; identifie le poursuivant |
| `hiding` | un résident habituel invisible depuis 45 s |
| `new_resident`, `return`, `record` | nouveau venu, retour après absence, record de vitesse |

L'« **humeur** » (serein, joueur, excité, endormi, sur le qui-vive, festin, désert) résume tout
ça en un mot pour le stream, et une petite narration (« Ce que je remarque ») est générée.

## Données

Tout est dans `outputs/ichtyodisco.sqlite` (une base, aucun serveur) :
`samples` (métriques de groupe, 2 Hz), `fish_samples` (positions 3D, vitesse, état par
poisson), `events`, et `kv` (le modèle appris). Ouvrable avec pandas, DB Browser, un notebook.
API JSON : `/api/snapshot`, `/api/history?minutes=60`, `/api/events`, `/api/residents`,
`/api/profile?metric=activity`, `/api/daily?days=7`, et le flux SSE `/api/live`.

## Son (OSC)

`--osc 127.0.0.1:57120` envoie `/aq/fish id x y z vitesse état` (≈20 Hz), `/aq/group`,
`/aq/mood` et `/aq/event`. Idées : z → réverbe / filtre passe-bas (les poissons du fond
sonnent « sous l'eau »), y → hauteur, vitesse → amplitude, écart à la normale → tension
harmonique, événements → samples one-shot.

## Entraîner le détecteur

Les poids COCO génériques ne connaissent pas « poisson ». Pour un vrai aquarium, un
fine-tuning sur ~200-500 images de *votre* cuve change tout :

```bash
uv run scripts/extract_frames.py enregistrement.mp4 datasets/fish/images --every 2
# annoter (Label Studio, CVAT, Roboflow) au format YOLO, une classe "fish"
uv run yolo detect train model=yolo11n.pt data=datasets/fish/data.yaml imgsz=640 epochs=80
cp runs/detect/train/weights/best.pt models/fish.pt
```

Conseils : varier éclairage / heures, inclure des poissons partiellement cachés et des
reflets sur la vitre (sinon les reflets deviennent des « poissons fantômes »).

## Développement

```bash
uv run pytest            # 30 tests, ~10 s, sans caméra ni GPU (simulateur)
uv run ruff check src tests
```

Arborescence :

```
src/aquarium/
  capture/     webcam (thread + reconnexion), fichiers vidéo / RTSP
  tracking/    YOLO + ByteTrack
  depth/       modèles HF, worker asynchrone, calibration cuve, échantillonnage par poisson
  analytics/   stats en ligne, cinématique, états, social, zones, résidents, événements, moteur
  storage/     SQLite (écriture asynchrone)
  render/      overlay broadcast
  stream/      dashboard web (stdlib), widget OBS, RTMP via ffmpeg
  sound/       pont OSC
  sim/         aquarium synthétique (démo + tests)
```

Les choix d'architecture sont détaillés dans `docs/decision-log.md`.

## Limites connues

- La profondeur monoculaire à travers une vitre reste **relative** : bonne pour « devant /
  milieu / fond », pas pour des centimètres. Pour du vrai métrique : deuxième caméra sur le
  côté (stéréo / triangulation) ou miroir à 45° au-dessus de la cuve.
- La ré-identification par couleur ne distingue pas des poissons identiques (banc de néons) :
  ils partagent alors quelques profils. Un modèle ReID appris ferait mieux.
- Les heures « apprises » demandent du temps réel : il faut ~1 h de données par tranche
  horaire, donc quelques jours de stream pour une journée complète.
