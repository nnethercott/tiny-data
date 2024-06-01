pub mod client;
pub mod download;
pub mod fetch;

use crate::client::*;
use pyo3::{prelude::*, wrap_pyfunction};

async fn main(topics: Vec<String>, nsamples: usize, dir: String) {
    let tiny_data_client = TinyDataClient::new();
    tiny_data_client.run(topics, nsamples, dir).await;
}

#[pyfunction]
fn run(py: Python<'_>, topics: Vec<String>, nsamples: usize, dir: String) -> PyResult<&PyAny> {
    pyo3_asyncio::tokio::future_into_py(py, async move {
        main(topics, nsamples, dir).await;
        Ok(Python::with_gil(|py| py.None()))
    })
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "tinydata")]
fn tokio_bindings(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    Ok(())
}
