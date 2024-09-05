#[warn(dead_code)]

use csv::ReaderBuilder;
use ndarray::{Array2, Array1, Axis, s};
use ndarray_rand::RandomExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::Deserialize;
use std::error::Error;




#[derive(Deserialize)]
struct Record {
    id: u32,
    diagnosis: String,
    features: [f64; 30],
}

pub struct LogisticRegressionGD {
    learning_rate: f64,
    n_iterations: usize,
    tolerance: f64,
    regularization: Option<String>,
    lambda: f64,
    theta: Array1<f64>,
    bias: f64,
}

impl LogisticRegressionGD {
    pub fn new(learning_rate: f64, n_iterations: usize, tolerance: f64, regularization: Option<String>, lambda: f64) -> Self {
        LogisticRegressionGD {
            learning_rate,
            n_iterations,
            tolerance,
            regularization,
            lambda,
            theta: Array1::zeros(30),
            bias: 0.0,
        }
    }

    fn sigmoid(&self, z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    pub fn fit(&mut self, X: &Array2<f64>, y: &Array1<f64>) {
        let (n_samples, n_features) = X.dim();
        self.theta = Array1::zeros(n_features);
        let mut previous_loss = f64::MAX;
    
        for iteration in 0..self.n_iterations {
            let linear_model = X.dot(&self.theta) + self.bias;
            let h: Array1<f64> = linear_model.mapv(|z| self.sigmoid(z));
    
            let mut dw = X.t().dot(&(h.clone() - y)) / n_samples as f64;
            let db = (h.clone() - y).sum() / n_samples as f64;
    
            // Regularization
            if let Some(ref reg) = self.regularization {
                if reg == "ridge" {
                    dw += &((self.lambda / n_samples as f64) * &self.theta);
                } else if reg == "lasso" {
                    dw += &((self.lambda / n_samples as f64) * &self.theta.mapv(f64::signum));
                }
            }
    
            self.theta -= &(self.learning_rate * dw);
            self.bias -= self.learning_rate * db;
    
            // Calculate the loss for early stopping
            let loss = -y.dot(&h.mapv(|val| val.ln())) - (1.0 - y).dot(&h.mapv(|val| (1.0 - val).ln()));
            if (previous_loss - loss).abs() < self.tolerance {
                println!("Converged after {} iterations", iteration);
                break;
            }
            previous_loss = loss;
        }
    }
    

    pub fn predict(&self, X: &Array2<f64>) -> Array1<u8> {
        let linear_model = X.dot(&self.theta) + self.bias;
        linear_model.mapv(|z| if self.sigmoid(z) > 0.5 { 1 } else { 0 })
    }
}

pub fn k_fold_cross_validation(model: &mut LogisticRegressionGD, X: &Array2<f64>, y: &Array1<f64>, k: usize) -> f64 {
    let mut rng = StdRng::seed_from_u64(42);
    let n_samples = X.len_of(Axis(0));
    let indices = ndarray::Array1::random_using(n_samples, rand::distributions::Uniform::new(0, n_samples), &mut rng);
    let mut accuracies = Vec::with_capacity(k);

    for i in 0..k {
        let validation_indices: Vec<usize> = indices.slice(s![i * n_samples / k..(i + 1) * n_samples / k]).to_vec();
        let train_indices: Vec<usize> = indices.slice(s![..i * n_samples / k])
            .to_vec().into_iter().chain(validation_indices.clone()).collect();

        let X_train = X.select(Axis(0), &train_indices);
        let y_train = y.select(Axis(0), &train_indices);
        let X_val = X.select(Axis(0), &validation_indices);
        let y_val = y.select(Axis(0), &validation_indices);

        model.fit(&X_train, &y_train);
        let predictions = model.predict(&X_val);
        let accuracy = predictions.iter().zip(y_val.iter()).filter(|&(p, t)| *p as f64 == *t).count() as f64 / y_val.len() as f64;
        accuracies.push(accuracy);
    }

    accuracies.iter().copied().sum::<f64>() / accuracies.len() as f64
}

pub fn load_data(file_path: &str) -> Result<(Array2<f64>, Array1<f64>), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(false).from_path(file_path)?;
    let mut X = Vec::new();
    let mut y = Vec::new();

    // Caricamento dei dati dal file
    for result in rdr.deserialize() {
        let record: (String, String, [f64; 30]) = result?;
        X.push(record.2.to_vec());
        y.push(if record.1 == "M" { 1.0 } else { 0.0 }); // Encoding delle classi
    }

    let mut X_array = Array2::from_shape_vec((X.len(), 30), X.into_iter().flatten().collect())?;
    let y_array = Array1::from_vec(y);

    // Imputazione dei NaN con la media delle colonne
    for mut col in X_array.gencolumns_mut() {
        let mean = col.mean().unwrap_or(0.0);
        col.iter_mut().for_each(|x| {
            if x.is_nan() {
                *x = mean;
            }
        });
    }

    // Normalizzazione delle feature
    let mean = X_array.mean_axis(Axis(0)).unwrap();
    let stddev = X_array.std_axis(Axis(0), 0.0);
    X_array = (&X_array - &mean) / &stddev;

    Ok((X_array, y_array))
}


