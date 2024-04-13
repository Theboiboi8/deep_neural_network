use std::collections::HashMap;

use ndarray::{Array, Array2};
use rand::distributions::{Distribution, Uniform};

use crate::data::{
    linear_backward_activation, linear_forward_activation, ActivationCache, LinearCache, Log,
};

pub struct DeepNeuralNetwork {
    pub layers: Vec<usize>,
    pub learning_rate: f32,
}

impl DeepNeuralNetwork {
    /// Initializes the parameters of the neural network.
    ///
    /// ### Returns
    /// a Hashmap dictionary of randomly initialized weights and biases.
    pub fn initialize_parameters(&self) -> HashMap<String, Array2<f32>> {
        let between = Uniform::try_from(-1.0..1.0);
        let mut rng = rand::thread_rng();

        let number_of_layers = self.layers.len();

        let mut parameters: HashMap<String, Array2<f32>> = HashMap::new();

        for layer in 1..number_of_layers {
            let weight_array: Vec<f32> = (0..self.layers[layer] * self.layers[layer - 1])
                .map(|_| between.unwrap().sample(&mut rng))
                .collect();

            let bias_array: Vec<f32> = (0..self.layers[layer]).map(|_| 0.0).collect();

            let weight_matrix =
                Array::from_shape_vec((self.layers[layer], self.layers[layer - 1]), weight_array)
                    .unwrap();
            let bias_matrix = Array::from_shape_vec((self.layers[layer], 1), bias_array).unwrap();

            let weight_string = ["W", &layer.to_string()].join("").to_string();
            let biases_string = ["b", &layer.to_string()].join("").to_string();

            parameters.insert(weight_string, weight_matrix);
            parameters.insert(biases_string, bias_matrix);
        }

        parameters
    }

    pub fn update_parameters(
        &self,
        params: &HashMap<String, Array2<f32>>,
        grads: HashMap<String, Array2<f32>>,
        learning_rate: f32,
    ) -> HashMap<String, Array2<f32>> {
        let mut parameters = params.clone();
        let num_of_layers = self.layers.len() - 1;
        for l in 1..=num_of_layers {
            let weight_string_grad = ["dW", &l.to_string()].join("").to_string();
            let bias_string_grad = ["db", &l.to_string()].join("").to_string();
            let weight_string = ["W", &l.to_string()].join("").to_string();
            let bias_string = ["b", &l.to_string()].join("").to_string();

            *parameters.get_mut(&weight_string).unwrap() = parameters[&weight_string].clone()
                - (learning_rate * (grads[&weight_string_grad].clone()));
            *parameters.get_mut(&bias_string).unwrap() = parameters[&bias_string].clone()
                - (learning_rate * grads[&bias_string_grad].clone());
        }
        parameters
    }

    pub fn train_model(
        &self,
        x_train_data: &Array2<f32>,
        y_train_data: &Array2<f32>,
        mut parameters: HashMap<String, Array2<f32>>,
        iterations: usize,
        learning_rate: f32,
    ) -> HashMap<String, Array2<f32>> {
        let mut costs: Vec<f32> = vec![];

        for i in 0..iterations {
            let (al, caches) = self.forward(x_train_data, &parameters);
            let cost = self.cost(&al, y_train_data);
            let grads = self.backward(&al, y_train_data, caches);
            parameters = self.update_parameters(&parameters, grads.clone(), learning_rate);

            if i % 100 == 0 {
                costs.append(&mut vec![cost]);
                println!("Epoch: {i:<5}/{iterations} | Cost: {cost:?}");
            }
        }
        parameters
    }

    pub fn predict(
        &self,
        x_test_data: &Array2<f32>,
        parameters: &HashMap<String, Array2<f32>>,
    ) -> Array2<f32> {
        let (al, _) = self.forward(x_test_data, parameters);

        let y_hat = al.map(|x| i32::from(x > &0.5) as f32);
        y_hat
    }

    pub fn score(&self, y_hat: &Array2<f32>, y_test_data: &Array2<f32>) -> f32 {
        let error =
            (y_hat - y_test_data).map(|x| x.abs()).sum() / y_test_data.shape()[1] as f32 * 100.0;
        100.0 - error
    }

    pub fn forward(
        &self,
        x: &Array2<f32>,
        parameters: &HashMap<String, Array2<f32>>,
    ) -> (Array2<f32>, HashMap<String, (LinearCache, ActivationCache)>) {
        let number_of_layers = self.layers.len() - 1;

        let mut a = x.clone();
        let mut caches = HashMap::new();

        for l in 1..number_of_layers {
            let w_string = ["W", &l.to_string()].join("").to_string();
            let b_string = ["b", &l.to_string()].join("").to_string();

            let w = &parameters[&w_string];
            let b = &parameters[&b_string];

            let (a_temp, cache_temp) = linear_forward_activation(&a, w, b, "relu").unwrap();

            a = a_temp;

            caches.insert(l.to_string(), cache_temp);
        }

        // Compute activation of last layer with sigmoid
        let weight_string = ["W", &number_of_layers.to_string()].join("").to_string();
        let bias_string = ["b", &number_of_layers.to_string()].join("").to_string();

        let w = &parameters[&weight_string];
        let b = &parameters[&bias_string];

        let (al, cache) = linear_forward_activation(&a, w, b, "sigmoid").unwrap();
        caches.insert(number_of_layers.to_string(), cache);

        (al, caches)
    }

    pub fn backward(
        &self,
        al: &Array2<f32>,
        y: &Array2<f32>,
        caches: HashMap<String, (LinearCache, ActivationCache)>,
    ) -> HashMap<String, Array2<f32>> {
        let mut grads = HashMap::new();
        let num_of_layers = self.layers.len() - 1;

        let dal = -(y / al - (1.0 - y) / (1.0 - al));

        let current_cache = caches[&num_of_layers.to_string()].clone();
        let (mut da_prev, mut dw, mut db) =
            linear_backward_activation(&dal, current_cache, "sigmoid");

        let weight_string = ["dW", &num_of_layers.to_string()].join("").to_string();
        let bias_string = ["db", &num_of_layers.to_string()].join("").to_string();
        let activation_string = ["dA", &num_of_layers.to_string()].join("").to_string();

        grads.insert(weight_string, dw);
        grads.insert(bias_string, db);
        grads.insert(activation_string, da_prev.clone());

        for l in (1..num_of_layers).rev() {
            let current_cache = caches[&l.to_string()].clone();
            (da_prev, dw, db) = linear_backward_activation(&da_prev, current_cache, "relu");

            let weight_string = ["dW", &l.to_string()].join("").to_string();
            let bias_string = ["db", &l.to_string()].join("").to_string();
            let activation_string = ["dA", &l.to_string()].join("").to_string();

            grads.insert(weight_string, dw);
            grads.insert(bias_string, db);
            grads.insert(activation_string, da_prev.clone());
        }

        grads
    }

    pub fn cost(&self, al: &Array2<f32>, y: &Array2<f32>) -> f32 {
        let m = y.shape()[1] as f32;
        let cost = -(1.0 / m)
            * (y.dot(&al.clone().reversed_axes().log())
            + (1.0 - y).dot(&(1.0 - al).reversed_axes().log()));

        cost.sum()
    }
}
