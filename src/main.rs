mod data;
mod neural_network;

fn main() {
    let neural_network_layers: Vec<usize> = vec![12288, 20, 7, 5, 1];
    let learning_rate = 0.025;
    let iterations = 10000;

    let (training_data, training_labels) =
        data::dataframe_from_csv("data/training_set.csv".into()).unwrap();
    let (test_data, test_labels) = data::dataframe_from_csv("data/test_set.csv".into()).unwrap();

    let training_data_array = data::array_from_dataframe(&training_data) / 255.0;
    let training_labels_array = data::array_from_dataframe(&training_labels);
    let test_data_array = data::array_from_dataframe(&test_data) / 255.0;
    let test_labels_array = data::array_from_dataframe(&test_labels);

    let model = neural_network::DeepNeuralNetwork {
        layers: neural_network_layers,
        learning_rate,
    };

    let parameters = model.initialize_parameters();

    let parameters = model.train_model(
        &training_data_array,
        &training_labels_array,
        parameters,
        iterations,
        model.learning_rate,
    );
    data::write_parameters_to_json_file(&parameters, "model.json".into());

    let training_predictions = model.predict(&training_data_array, &parameters);
    println!(
        "Training Set Accuracy: {}%",
        model.score(&training_predictions, &training_labels_array)
    );

    let test_predictions = model.predict(&test_data_array, &parameters);
    println!(
        "Test Set Accuracy: {}%",
        model.score(&test_predictions, &test_labels_array)
    );
}
