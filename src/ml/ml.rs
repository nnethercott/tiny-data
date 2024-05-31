use anyhow::Error as E;

use candle_core::{shape::Dims, DType, Device, Tensor};
use candle_nn::{ops::softmax, VarBuilder};
use candle_transformers::models::clip;
use tokenizers::Tokenizer;

//TODO: convert to using onnx

//combines a bit of preprocessing
fn load_image<T: AsRef<std::path::Path>>(path: T, image_size: usize) -> anyhow::Result<Tensor> {
    let img = image::io::Reader::open(path)?.decode()?;
    let (height, width) = (image_size, image_size);
    let img = img.resize_to_fill(
        width as u32,
        height as u32,
        image::imageops::FilterType::Triangle,
    );

    let img = img.to_rgb8();
    let dtype = DType::F32;

    //https://github.com/huggingface/candle/blob/main/candle-examples/src/imagenet.rs#L5
    let mean =
        Tensor::new(&[0.48145466f32, 0.4578275, 0.40821073], &Device::Cpu)?.reshape((3, 1, 1))?;
    let std =
        Tensor::new(&[0.26862954f32, 0.26130258, 0.27577711], &Device::Cpu)?.reshape((3, 1, 1))?;

    let data = img.into_raw();
    let data = Tensor::from_vec(data, (224, 224, 3), &Device::Cpu)?.permute((2, 0, 1))?;

    let data = (data.to_dtype(dtype)? / 255.)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)?;

    Ok(data)
}

fn load_images<T: AsRef<std::path::Path>>(
    paths: &Vec<T>,
    image_size: usize,
) -> anyhow::Result<Tensor> {
    let mut images = vec![]; //Vec<Tensor>

    for path in paths {
        let tensor = load_image(path, image_size)?;
        images.push(tensor)
    }

    let images = Tensor::stack(&images, 0)?;

    Ok(images)
}

//huggingface api
fn get_model_and_tokenizer() -> anyhow::Result<(clip::ClipModel, clip::ClipConfig, Tokenizer)> {
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

fn tokenize_sequences(
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
fn norm2(t: Tensor, device: &Device, _dtype: Option<DType>) -> anyhow::Result<Tensor> {
    let norm = t.sqr()?.sum(1)?.sqrt()?.unsqueeze(1)?.repeat((1, 512))?; //clip seq len
    let normalized = (t / norm)?;

    Ok(normalized)
}

fn main() -> anyhow::Result<()> {
    //paths to some images
    let (model, config, tokenizer) = get_model_and_tokenizer()?;

    //tokenizer descriptions and encode images
    let device = Device::Cpu;
    let images = load_images(
        &vec![
            "./images/cats/3.jpeg",
            "./images/cats/2.jpeg",
            "./images/dogs/0.jpeg",
            "./images/dogs/5.jpeg",
            "./images/dogs/1.jpeg",
        ],
        config.image_size,
    )?;
    let seqs = vec!["a dog playing outdoors".to_string()];
    let (input_ids, _) = tokenize_sequences(seqs, &tokenizer, &device)?;

    let text_features = model.get_text_features(&input_ids)?;
    let img_features = model.get_image_features(&images)?;

    let img_features = norm2(img_features, &device, None)?;
    let text_features = norm2(text_features, &device, None)?;

    // println!("{:?}", text_features.sqr()?.sum_all()?.sqrt()?);

    let logits = img_features
        .matmul(&text_features.transpose(0, 1)?)?
        .flatten_all()?;

    println!("{:?}", logits.to_vec1::<f32>()?);

    // let logits = text_features.matmul(&img_features.transpose(0, 1)?)?;
    // println!("{:?}", logits.to_vec2::<f32>());

    Ok(())
}
