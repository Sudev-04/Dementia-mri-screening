# Models

- `demo/` is a three-model ensemble trained for a few epochs on **synthetic noise volumes**
  (`python -m ml.synthetic`). It exists so the clinic page can be tried end to end. Its scores
  are meaningless, and the app labels every output from it as a demo. It is only loaded when
  the server is started with `--demo-model`.
- `current/` is where a real export goes (`python -m ml.export --run runs/<name> --dest models/current`).
  It is git-ignored: check the OASIS data use terms before publishing weights trained on OASIS data.
