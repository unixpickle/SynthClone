const TOTAL_TOKENS = (1024 * 5 * 24) >> 8;

class App {
    constructor() {
        this.request = null;
        this.currentTokens = null;

        this.error = document.getElementById('error');
        this.prompt = document.getElementById('prompt');
        this.generateButton = document.getElementById('generate-button');
        this.generateButton.addEventListener('click', () => {
            this.generateOrCancel();
        });
        this.outputContainer = document.getElementById('output');
        this.sampledAudio = document.getElementById('sampled-audio');
        this.sampledAudioSource = document.getElementById('sampled-audio-source');
        this.progress = document.getElementById('loader-contents');
        this.guidanceScale = document.getElementById('guidance-scale');
        this.lowpassCutoff = document.getElementById('lowpass-cutoff');

        this.prompt.addEventListener('keypress', (event) => {
            if (event.key === "Enter") {
                event.preventDefault();
                this.cancel();
                this.generate();
            }
        });
    }

    generateOrCancel() {
        if (this.request) {
            this.cancel();
        } else {
            this.generate();
        }
    }

    cancel() {
        if (!this.request) {
            return;
        }
        this.request.cancel();
        this.request = null;
        this.generateButton.classList.remove('cancel');
        this.generateButton.textContent = 'Generate';
        this.outputContainer.className = '';
    }

    generate() {
        this.generateButton.classList.add('cancel');
        this.generateButton.textContent = 'Cancel';
        this.currentTokens = [];
        this.request = new SamplingRequest(this.prompt.value, this.guidanceScale.value);
        this.request.onError = (err) => this.showError(err);
        this.request.onToken = (token) => this.handleToken(token);
        this.request.onDone = () => this.cancel();
        this.request.start();
        this.outputContainer.className = 'loading';
        this.updateProgress();
    }

    showError(err) {
        this.outputContainer.className = 'error';
        this.error.textContent = err;
    }

    handleToken(token) {
        this.currentTokens.push(token);
        const isDone = this.currentTokens.length == TOTAL_TOKENS;
        this.outputContainer.className = isDone ? 'done' : 'loading';
        if (this.currentTokens.length % 96 == 0 || isDone) {
            this.sampledAudioSource.src = (
                '/decode?lowpassCutoff=' + this.lowpassCutoff.value +
                '&tokens=' + this.currentTokens.join(',')
            );
            this.sampledAudio.load();
        }
        this.updateProgress();
    }

    updateProgress() {
        this.progress.style.width = (100 * this.currentTokens.length / TOTAL_TOKENS).toFixed(2) + '%';
    }
}

class SamplingRequest {
    constructor(prompt, guidanceScale) {
        this.url = ('/sample?guidanceScale=' + guidanceScale +
            '&prompt=' + encodeURIComponent(prompt));
        this.onToken = (t) => null;
        this.onDone = () => null;
        this.onError = (e) => null;
        this._controller = null;
        this._lastTokens = [];
    }

    start() {
        const xhr = new XMLHttpRequest();
        xhr.open("GET", this.url);
        xhr.responseType = "text";

        xhr.onprogress = () => {
            if (xhr.readyState === XMLHttpRequest.LOADING || xhr.readyState == XMLHttpRequest.DONE) {
                if (xhr.status >= 200 && xhr.status < 300) {
                    this._processBuffer(xhr.responseText);
                    if (xhr.readyState == XMLHttpRequest.DONE) {
                        this.onDone();
                    }
                } else {
                    this.onError('invalid status code: ' + xhr.status);
                    this.cancel();
                }
            }
        };
        xhr.onreadystatechange = xhr.onprogress;
        xhr.onerror = () => this.onError('failed to fetch results');

        this._controller = xhr;
        xhr.send();
    }

    _processBuffer(buffer) {
        let tokens = [];
        while (true) {
            const newlineIndex = buffer.indexOf("\n");
            if (newlineIndex < 0) {
                break;
            }
            const integerString = buffer.slice(0, newlineIndex);
            buffer = buffer.slice(newlineIndex + 1);

            // Ignore blank lines.
            if (integerString.length == 0) {
                continue;
            }

            tokens.push(parseInt(integerString, 10));
        }
        for (let i = this._lastTokens.length; i < tokens.length; i++) {
            this.onToken(tokens[i]);
        }
        this._lastTokens = tokens;
    }

    cancel() {
        if (this._controller) {
            this._controller.abort();
            this._controller = null;
        }
    }
}

window.app = new App();