"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var transformers_1 = require("@huggingface/transformers");
var test = function () {
    (0, transformers_1.pipeline)("automatic-speech-recognition", "openai/whisper-large-v3", {
        progress_callback: function (progress) {
            console.log(progress.status);
        },
        dtype: "fp16",
    });
};
