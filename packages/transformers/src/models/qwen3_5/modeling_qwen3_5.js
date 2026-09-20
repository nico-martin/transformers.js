import { Qwen3VLForConditionalGeneration } from '../qwen3_vl/modeling_qwen3_vl.js';
import { PreTrainedModel } from '../modeling_utils.js';
import { SequenceClassifierOutput } from '../modeling_outputs.js';

export class Qwen3_5ForConditionalGeneration extends Qwen3VLForConditionalGeneration {}
export class Qwen3_5ForCausalLM extends Qwen3_5ForConditionalGeneration {}

/**
 * Qwen3.5 text model with a sequence classification head on top (a linear layer over the
 * hidden state of the last non-padding token). Used for scorers and reward models exported
 * as a single `model.onnx` graph with `input_ids` and `attention_mask` inputs and a `logits` output.
 */
export class Qwen3_5ForSequenceClassification extends PreTrainedModel {
    /**
     * Calls the model on new inputs.
     *
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
