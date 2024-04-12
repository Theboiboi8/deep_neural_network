use std::collections::HashMap;
use std::f32::consts::E;
use std::fs::OpenOptions;
use std::path::PathBuf;

use ndarray::prelude::*;
use polars::prelude::*;

pub fn write_parameters_to_json_file(
    parameters: &HashMap<String, Array2<f32>>,
    file_path: PathBuf,
) {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(file_path)
        .unwrap();

    _ = serde_json::to_writer(file, parameters);
}

#[allow(clippy::missing_errors_doc)]
pub fn dataframe_from_csv(file_path: PathBuf) -> PolarsResult<(DataFrame, DataFrame)> {
    let data = CsvReader::from_path(file_path)?.has_header(true).finish()?;

    let training_dataset = data.drop("y")?;
    let training_labels = data.select(["y"])?;

    Ok((training_dataset, training_labels))
}

pub fn array_from_dataframe(data_frame: &DataFrame) -> Array2<f32> {
    data_frame
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap()
        .reversed_axes()
}

#[derive(Clone, Debug)]
pub struct LinearCache {
    pub a: Array2<f32>,
    pub w: Array2<f32>,
    pub b: Array2<f32>,
}

#[derive(Clone, Debug)]
pub struct ActivationCache {
    pub z: Array2<f32>,
}

pub fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + E.powf(-z))
}

/// Rectified Linear Unit (`ReLU`)
pub fn relu(z: f32) -> f32 {
    if z > 0.0 {
        z
    } else {
        0.0
    }
}

pub fn sigmoid_activation(z: Array2<f32>) -> (Array2<f32>, ActivationCache) {
    (z.mapv(sigmoid), ActivationCache { z })
}

pub fn relu_activation(z: Array2<f32>) -> (Array2<f32>, ActivationCache) {
    (z.mapv(relu), ActivationCache { z })
}

pub fn linear_forward(
    a: &Array2<f32>,
    w: &Array2<f32>,
    b: &Array2<f32>,
) -> (Array2<f32>, LinearCache) {
    let z = w.dot(a) + b;

    let cache = LinearCache {
        a: a.clone(),
        w: w.clone(),
        b: b.clone(),
    };

    (z, cache)
}

pub fn linear_forward_activation(
    a: &Array2<f32>,
    w: &Array2<f32>,
    b: &Array2<f32>,
    activation: &str,
) -> Result<(Array2<f32>, (LinearCache, ActivationCache)), String> {
    match activation {
        "sigmoid" => {
            let (z, linear_cache) = linear_forward(a, w, b);
            let (a_next, activation_cache) = sigmoid_activation(z);

            Ok((a_next, (linear_cache, activation_cache)))
        }
        "relu" => {
            let (z, linear_cache) = linear_forward(a, w, b);
            let (a_next, activation_cache) = relu_activation(z);

            Ok((a_next, (linear_cache, activation_cache)))
        }
        _ => Err("wrong activation string".to_string()),
    }
}

pub fn sigmoid_prime(z: f32) -> f32 {
    sigmoid(z) * (1.0 - sigmoid(z))
}

pub fn relu_prime(z: f32) -> f32 {
    if z > 0.0 {
        1.0
    } else {
        0.0
    }
}

pub fn sigmoid_backward(da: &Array2<f32>, activation_cache: ActivationCache) -> Array2<f32> {
    da * activation_cache.z.mapv(sigmoid_prime)
}

pub fn relu_backward(da: &Array2<f32>, activation_cache: ActivationCache) -> Array2<f32> {
    da * activation_cache.z.mapv(relu_prime)
}

pub fn linear_backward(
    dz: &Array2<f32>,
    linear_cache: LinearCache,
) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let (a_prev, w, _b) = (linear_cache.a, linear_cache.w, linear_cache.b);
    let m = a_prev.shape()[1] as f32;
    let dw = (1.0 / m) * (dz.dot(&a_prev.reversed_axes()));
    let db_vec = ((1.0 / m) * dz.sum_axis(Axis(1))).to_vec();
    let db = Array2::from_shape_vec((db_vec.len(), 1), db_vec).unwrap();
    let da_prev = w.reversed_axes().dot(dz);

    (da_prev, dw, db)
}

pub fn linear_backward_activation(
    da: &Array2<f32>,
    cache: (LinearCache, ActivationCache),
    activation: &str,
) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let (linear_cache, activation_cache) = cache;

    match activation {
        "sigmoid" => {
            let dz = sigmoid_backward(da, activation_cache);
            linear_backward(&dz, linear_cache)
        }
        "relu" => {
            let dz = relu_backward(da, activation_cache);
            linear_backward(&dz, linear_cache)
        }
        _ => panic!("wrong activation string"),
    }
}

pub trait Log {
    fn log(&self) -> Array2<f32>;
}

impl Log for Array2<f32> {
    fn log(&self) -> Array2<f32> {
        self.mapv(|x| x.log(E))
    }
}
