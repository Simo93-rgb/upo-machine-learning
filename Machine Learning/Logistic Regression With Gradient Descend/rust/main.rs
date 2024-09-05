use std::time::Instant;
use ndarray::{Array2, Array1, Axis};
mod logistic_regression_gradient_descend;
use logistic_regression_gradient_descend::{k_fold_cross_validation, load_data, LogisticRegressionGD};

fn train_test_split(X: &Array2<f64>, y: &Array1<f64>, test_size: f64) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>) {
    let n_samples = X.nrows();
    let n_train = (n_samples as f64 * (1.0 - test_size)).round() as usize;

    let (X_train, X_val) = X.view().split_at(Axis(0), n_train);
    let (y_train, y_val) = y.view().split_at(Axis(0), n_train);

    (X_train.to_owned(), X_val.to_owned(), y_train.to_owned(), y_val.to_owned())
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();

    // Caricamento e preprocessing dei dati
    let (X, y) = load_data("wdbc.data")?;
    
    // Split del dataset in training e validation set
    let (X_train, X_val, y_train, y_val) = train_test_split(&X, &y, 0.3);
    
    // Impostazione del modello
    let mut model = LogisticRegressionGD::new(0.05, 1000, 1e-10, Some("ridge".into()), 0.1);
    
    // Allenamento e valutazione su training e validation set
    model.fit(&X_train, &y_train);
    let val_accuracy = model.predict(&X_val).iter()
        .zip(y_val.iter())
        .filter(|&(p, t)| *p as f64 == *t)
        .count() as f64 / y_val.len() as f64;

    // Esecuzione della K-fold cross validation
    let k_fold_accuracy = k_fold_cross_validation(&mut model, &X, &y, 5);

    let duration = start.elapsed();
    
    // Stampa dei risultati
    println!("Validation accuracy: {:.4}", val_accuracy);
    println!("Mean accuracy (K-fold): {:.4}", k_fold_accuracy);
    println!("Time taken: {:?}", duration);

    Ok(())
}
