# ichtyodisco

Un aquarium qui s'observe lui-même et **se joue en musique**, en direct : tracking des
poissons, **profondeur monoculaire** (Apple Depth Pro / Depth Anything V2), analyse
comportementale qui **apprend la « normale » de l'aquarium heure par heure**, un **chef
d'orchestre MIDI** qui fait jouer tes synthés par les habitants, dashboard web et
**livestream RTMP** (YouTube / Twitch) avec le son des synthés.

```
caméra ──► YOLO + ByteTrack ──► profondeur (async) ──► moteur d'analyse ──► overlay vidéo ──► RTMP (YouTube/Twitch)
                                                          │                        └──────► dashboard web (MJPEG + SSE)
                                                          ├─► SQLite (séries, événements, modèle appris)
                                                          ├─► MIDI → tes synthés (hardware / DAW) ──► carte son ──► stream
                                                          └─► OSC (Max, SuperCollider, Pd, TouchDesigner)
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
   ralentit, le flux reste fluide — et redémarre tout seul si le réseau coupe. Pour mettre
   **le son de tes synthés** dans le stream, ajoute l'entrée de ta carte son :
   `--audio-in avfoundation::0` (Mac), `--audio-in pulse:default` (Linux),
   `--audio-in "dshow:audio=Line 1 (Focusrite)"` (Windows) ; sans ça, une piste muette est
   envoyée (YouTube l'exige). Sur GPU NVIDIA, mettre `encoder = "h264_nvenc"`.
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
| `depth-pro-2` (S, 24M) · `-b` · `-l` · `-h` (H+, 872M) | S temps réel sur GPU, H+ lent | Apple, **recherche / non commercial** | **le plus stable en direct** (mode streaming), usage perso |
| `depth-anything-v2-small` | rapide | Apache-2.0 | **défaut**, OK pour un stream public / monétisé |
| `depth-pro` (v1, Apple) | ~0,3 s/image sur gros GPU | Apple ASCL, recherche | qualité image par image |
| `depth-anything-v2-base/large` | moyen / lent | CC-BY-NC-4.0 | non commercial |

**Depth Pro 2** (Apple, novembre 2026) a un mode *streaming* : chaque image est prédite
avec les quelques images précédentes, donc la profondeur ne scintille plus d'une image à
l'autre — moins de faux mouvements en z, des vitesses 3D plus propres. Il donne aussi de la
profondeur **métrique** (en mètres). Il n'est pas sur PyPI, à installer à part :

```bash
uv pip install "git+https://github.com/apple-aiml-research/ml-depth-pro-2"
uv run ichtyodisco -s 0 --model models/fish.pt --depth depth-pro-2      # S, 97 Mo
uv run ichtyodisco -s 0 --depth depth-pro-2-l --device cuda:0             # plus net, plus lourd
```

Les poids se téléchargent au premier lancement. `[depth] window` règle le nombre d'images
passées prises en compte (4 par défaut : plus = plus stable, mais plus lent).

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

## Musique : l'aquarium joue tes synthés

```bash
uv sync --extra midi                     # ports MIDI temps réel (python-rtmidi)
uv run ichtyodisco -s 0 --midi           # crée un port MIDI virtuel « ichtyodisco »
uv run ichtyodisco -s 0 --midi IAC       # ou un port existant : IAC (Mac), loopMIDI (Windows), ton synthé USB
uv run ichtyodisco --demo --midi --midi-record outputs/session.mid --key F# --bpm 90
uv run ichtyodisco --midi-list           # lister les ports
```

Un **chef d'orchestre** génératif traduit le comportement en musique, toujours dans la
tonalité (rien ne sonne faux), et la musique évolue avec ce que le modèle apprend :

| aquarium | musique |
|---|---|
| humeur (serein, joueur, excité, endormi, qui-vive, festin, désert) | mode et progression d'accords : lydien flottant quand c'est serein, dorien quand ça s'agite, phrygien tendu en alerte, majeur pendant le repas, drone quand l'aquarium est vide |
| activité vs normale de l'heure | tempo (±), rythme de la basse, densité de la batterie |
| cohésion du banc | voicing du pad : banc serré = accord serré, banc dispersé = accord ouvert |
| polarisation | molette de modulation (CC1) du pad |
| **chaque poisson = une voix / un canal MIDI** | hauteur = profondeur verticale (surface = aigu) ; rythme = état de nage (repos = notes longues et rares, croisière = croches, sprint = rafales de doubles) ; vélocité = vitesse ; pan (CC10) = gauche-droite ; **brillance CC74 et réverbe CC91 = distance à la vitre** (un poisson au fond sonne « lointain ») |
| événements | phrases : repas = arpège montant + crash, poursuite = question-réponse entre les voix des deux poissons, nouveau venu = cloche, record = gamme éclair, alerte = trémolo de seconde mineure |

Canaux par défaut : **1** pad, **2** basse, **3** lead/événements, **4-9** poissons, **10**
batterie (mapping General MIDI : kick 36, rim 37, hats 42/46, crash 49, shaker 82). Dans
Ableton / Bitwig / Logic : une piste par canal, chacune avec l'instrument de ton choix
(Vital, Diva, Pigments, un sampler…). En hardware : règle le canal MIDI de chaque synthé.
`send_clock = true` envoie l'horloge MIDI pour que les arpégiateurs et le DAW suivent le
tempo de l'aquarium. Tout est réglable dans `[music]` (`config/example.toml`).

`--midi-record` enregistre la session en `.mid` (timing réel), pour la retravailler dans un DAW.

> `python-rtmidi` fournit des binaires jusqu'à Python 3.12 ; en 3.13 il se compile (Xcode CLT
> sur Mac, `libasound2-dev` sur Linux, MSVC sur Windows). Le plus simple : `uv python pin 3.12`.

### Mode partition « jeu vidéo » (`--score`)

```bash
uv run ichtyodisco -s 0 --midi --score scores/demo/score.toml
uv run ichtyodisco --demo --midi --score scores/demo/score.toml --midi-record session.mid
```

Ici, c'est **toi qui composes** (des clips MIDI dans ton DAW) et ce sont les poissons qui
décident de ce qui joue, comme une musique de jeu adaptative (iMUSE / Wwise / FMOD) :

- **Sections** : l'état de l'aquarium choisit la section (calme, exploration, banc,
  festin, nuit, vide…) via des conditions (`when = "polarization > 0.75 and fish_count >= 3"`)
  et des priorités. Les changements attendent un temps, une mesure ou une phrase, et peuvent
  passer par un **pont** propre à chaque transition (`calm -> feeding` ≠ `feeding -> *`).
- **Couches** : chaque section est une pile de boucles dont le volume suit les métriques en
  direct. Les **overlays** vivent hors des sections, et une couche partagée entre deux
  sections continue sans repartir de zéro : c'est là que les sections se croisent.
- **Stingers** : les **interactions fines** déclenchent des clips ou une simple note (un
  sample dans ton sampler), calés sur la grille, panoramiqués là où ça s'est passé,
  transposés / dosés selon l'événement. Exemple de la démo : un poisson sort du banc une
  seconde et revient → *spike* (`note 76`, canal 10), plus fort s'il revient vite.
- **Signaux** : des jauges d'intensité nourries par les événements, qui retombent toutes
  seules (`excursions`, `chase_heat`) et pilotent n'importe quelle couche.
- **Harmonie** : les clips sont écrits dans un ton et un mode (`[harmony] key = "D"`,
  `written_in = "dorian"`). À chaque mesure, l'aquarium choisit le **mode** joué (lydien
  quand c'est serein et haut, phrygien en alerte, éolien la nuit…) et peut **décaler la
  tonique** (+1 ton pendant le repas) : chaque note garde son degré, les mêmes clips changent
  de couleur. `degrees` décale une couche ou un stinger **dans la gamme** (la phrase monte
  quand le poisson revient près de la surface, les arpèges suivent la hauteur du banc).
- **Modulations** : CC et pitch bend continus pilotés par la position et le comportement
  (`[[modulation]]` : brillance CC74 = proximité de la vitre, molette CC1 = polarisation,
  réverbe CC91 = profondeur, pan CC10 = position gauche-droite du banc, pitch bend = hauteur
  du poisson le plus rapide), avec lissage. Plus le **swing** (`swing = "0.18 if mood ==
  'joueur' else 0"`) et le **tempo** en expressions.

Dans les expressions, en plus des métriques : la géométrie du banc (`centroid_x/y/z`,
`spread`) et du poisson le plus rapide (`lead_x/y/z`, `lead_speed`).

Interactions détectées image par image (relatives à l'espacement du banc) :
`school_leave`, `school_return` (+ durée de l'escapade), `school_split`, `school_merge`,
`encounter`, `dash`, `surface`, `bottom`, `glass` (un poisson vient à la vitre),
`back_wall`, `group_turn`, `appear`, `vanish` — en plus des événements lents (`feeding`,
`chase`, `hiding`, `new_resident`…). Chacune porte `x y z speed` (et `duration`,
`distance`, `groups`, `angle` selon le cas), utilisables dans les expressions.

Tout est décrit dans `scores/demo/score.toml` (commenté). Les clips de la démo sont des
placeholders générés par `scripts/make_demo_score.py` : remplace-les par les tiens. Les
expressions sont un mini-langage sûr (arithmétique, comparaisons, `clip`, `smoothstep`…) :
un fichier de partition ne peut pas exécuter de code.

### OSC (Max/MSP, SuperCollider, TouchDesigner…)

`--osc 127.0.0.1:57120` envoie `/aq/fish id x y z vitesse état` (≈20 Hz), `/aq/group`,
`/aq/mood` et `/aq/event` (événements lents *et* interactions fines : `school_return`,
`encounter`…). Dans Max : `[udpreceive 57120]` → `[route /aq/fish /aq/group /aq/event]`.
Pratique pour des visuels réactifs ou des modulations en plus du MIDI.

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
uv run pytest            # 57 tests, ~10 s, sans caméra ni GPU (simulateur)
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
  music/       musique MIDI : chef d'orchestre génératif, partition adaptative (sections,
               couches, stingers), expressions, clips, ports / enregistrement .mid
scores/demo/   partition de démo (score.toml + clips .mid)
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
