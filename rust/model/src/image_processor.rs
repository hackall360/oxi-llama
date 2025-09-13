use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, Pixel, Rgba, RgbaImage};

pub const RESIZE_BILINEAR: u32 = 0;
pub const RESIZE_NEAREST_NEIGHBOR: u32 = 1;
pub const RESIZE_APPROX_BILINEAR: u32 = 2; // uses Triangle
pub const RESIZE_CATMULLROM: u32 = 3;

pub fn composite(img: &DynamicImage) -> DynamicImage {
    let mut dst = RgbaImage::new(img.width(), img.height());
    for (_x, _y, pixel) in dst.enumerate_pixels_mut() {
        *pixel = Rgba([255, 255, 255, 255]);
    }
    image::imageops::overlay(&mut dst, img, 0, 0);
    DynamicImage::ImageRgba8(dst)
}

pub fn resize(img: &DynamicImage, new_width: u32, new_height: u32, method: u32) -> DynamicImage {
    let filter = match method {
        RESIZE_BILINEAR => FilterType::Triangle,
        RESIZE_NEAREST_NEIGHBOR => FilterType::Nearest,
        RESIZE_APPROX_BILINEAR => FilterType::Triangle,
        RESIZE_CATMULLROM => FilterType::CatmullRom,
        _ => panic!("no resizing method found"),
    };
    let resized = image::imageops::resize(img, new_width, new_height, filter);
    DynamicImage::ImageRgba8(resized)
}

pub fn normalize(
    img: &DynamicImage,
    mean: [f32; 3],
    std: [f32; 3],
    rescale: bool,
    channel_first: bool,
) -> Vec<f32> {
    let (w, h) = img.dimensions();
    let mut values = Vec::new();
    if channel_first {
        let mut r_vals = Vec::new();
        let mut g_vals = Vec::new();
        let mut b_vals = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let p = img.get_pixel(x, y).to_rgba();
                let mut r = p[0] as f32;
                let mut g = p[1] as f32;
                let mut b = p[2] as f32;
                if rescale {
                    r /= 255.0;
                    g /= 255.0;
                    b /= 255.0;
                }
                r_vals.push((r - mean[0]) / std[0]);
                g_vals.push((g - mean[1]) / std[1]);
                b_vals.push((b - mean[2]) / std[2]);
            }
        }
        values.extend(r_vals);
        values.extend(g_vals);
        values.extend(b_vals);
    } else {
        for y in 0..h {
            for x in 0..w {
                let p = img.get_pixel(x, y).to_rgba();
                let mut r = p[0] as f32;
                let mut g = p[1] as f32;
                let mut b = p[2] as f32;
                if rescale {
                    r /= 255.0;
                    g /= 255.0;
                    b /= 255.0;
                }
                values.push((r - mean[0]) / std[0]);
                values.push((g - mean[1]) / std[1]);
                values.push((b - mean[2]) / std[2]);
            }
        }
    }
    values
}
