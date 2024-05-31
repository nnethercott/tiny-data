use anyhow::Error as E;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::clip;
use tokenizers::Tokenizer;

//TODO: convert to using onnx

//huggingface api
pub fn get_model_and_tokenizer() -> anyhow::Result<(clip::ClipModel, clip::ClipConfig, Tokenizer)> {
    let api = hf_hub::api::sync::Api::new()?;
    let api = api.repo(hf_hub::Repo::with_revision(
        "openai/clip-vit-base-patch32".to_string(),
        hf_hub::RepoType::Model,
        "refs/pr/15".to_string(),
    ));

    let model_file = api.get("model.safetensors")?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model_file.clone()], DType::F32, &Device::Cpu)?
    };

    let config = clip::ClipConfig::vit_base_patch32();
    let model = clip::ClipModel::new(vb, &config)?;

    let tokenizer_file = api.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(E::msg)?;

    Ok((model, config, tokenizer))
}

pub fn tokenize_sequences(
    sequences: Vec<String>,
    tokenizer: &Tokenizer,
    device: &Device,
) -> anyhow::Result<(Tensor, Vec<String>)> {
    let mut tokens = vec![];

    for seq in sequences.clone() {
        let encoding = tokenizer.encode(seq, true).map_err(E::msg)?;
        tokens.push(encoding.get_ids().to_vec());
    }
    let max_len: usize = tokens.iter().map(|v| v.len()).max().unwrap_or(0);

    //padding
    let pad_id = *tokenizer
        .get_vocab(true)
        .get("<|endoftext|>")
        .ok_or(E::msg("No pad token"))?;

    for tok in tokens.iter_mut() {
        let diff_len = max_len - tok.len();

        if diff_len > 0 {
            tok.extend(vec![pad_id; diff_len]); //cool use of vec! macro
        }
    }

    let input_ids = Tensor::new(tokens, device)?;

    Ok((input_ids, sequences))
}

//TODO: implement dtype stuff
pub fn norm2(t: Tensor, device: &Device, _dtype: Option<DType>) -> anyhow::Result<Tensor> {
    let norm = t.sqr()?.sum(1)?.sqrt()?.unsqueeze(1)?.repeat((1, 512))?; //clip seq len
    let normalized = (t / norm)?.to_device(&device)?;

    Ok(normalized)
}
