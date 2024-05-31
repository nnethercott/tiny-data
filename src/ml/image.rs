use candle_core::{DType, Device, Tensor};

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

pub fn load_images<T: AsRef<std::path::Path>>(
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
