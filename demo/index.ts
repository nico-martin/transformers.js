import { pipeline, ProgressInfo } from "@huggingface/transformers";
const test = () => {
  pipeline("automatic-speech-recognition", "openai/whisper-large-v3", {
    progress_callback: function (progress: ProgressInfo) {
      console.log(progress.status);
    },
    dtype: "fp16",
  });
};

