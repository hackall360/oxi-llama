# Desktop

This app builds upon Ollama to provide a desktop experience for running models.

## Developing

First, build the `ollama` binary:

```shell
cd ..
cargo build --release --bin ollama
```

Then run the desktop app with `npm start`:

```shell
cd macapp
npm install
npm start
```

