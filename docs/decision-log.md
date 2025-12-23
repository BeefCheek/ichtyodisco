## Décisions récentes

### Capture vidéo (déc. 2025)

- **Ce qui a été fait**
  - Création d’un wrapper `WebcamCapture` qui encapsule `cv2.VideoCapture`.
  - Gestion d’un thread dédié pour ne pas bloquer le pipeline principal.
  - Mise en place d’un buffer circulaire thread-safe (deque + verrou) afin de décorréler la cadence de capture de la cadence de consommation.
  - Tentatives automatiques de reconnexion quand `read()` échoue ou que la webcam se déconnecte.

- **Options discutées**
  - *Single-thread + `cap.read()` direct dans la loop principale* : rejeté car cela bloquait l’inférence et compliquait la régulation du framerate.
  - *Queue illimitée* : rejetée pour éviter les fuites mémoire en cas de backlog ; la deque bornée suffit car seuls les derniers frames importent.
  - *Reconnexion synchronisée côté appelant* : rejetée au profit d’une logique interne dans le thread, pour éviter que les consommateurs aient à gérer l’état matériel.

- **Raisons du choix actuel**
  - Le threading garantit un flux continu et prévisible pour le ML même si le traitement en aval varie.
  - Le buffer borné protège la mémoire et limite la latence.
  - La reconnexion automatique améliore la robustesse sans exposer la complexité au reste du code.
