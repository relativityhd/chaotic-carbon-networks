use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

mod lapend;
mod mind;

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "rust_chaotic_carbon_networks")]
fn chaotic_carbon_networks(_py: Python, m: &PyModule) -> PyResult<()> {
    /// Calculates the Mutual Information between every v in x of dimensions [v, t]. If a y is provided calculates the Mutual Information between every vx and vy of x [vx, t] and y [vy, t].
    #[pyfn(m, signature = (x, y, bins = 64))]
    #[pyo3(name = "mind")]
    fn mind_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
        y: Option<PyReadonlyArray2<'py, f32>>,
        bins: usize,
    ) -> &'py PyArray2<f32> {
        let x = x.as_array();
        let z = match y {
            Some(y_arr) => mind::mind_double(x, y_arr.as_array(), bins),
            None => mind::mind_single(x, bins),
        };
        z.into_pyarray(py)
    }

    /// Calculates the Lagged Pearson Correlation Coefficient between every  v in x of dimensions [v, t]. If a y is provided calculates the Lagged Pearson Correlation Coefficient between every vx and vy of x [vx, t] and y [vy, t].
    #[pyfn(m)]
    #[pyo3(name = "lapend")]
    fn lapend_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
        tau_min: isize,
        tau_max: isize,
        y: Option<PyReadonlyArray2<'py, f32>>,
    ) -> &'py PyArray2<f32> {
        let x = x.as_array();
        let z = match y {
            Some(y_arr) => lapend::lapend_double(x, y_arr.as_array(), tau_min, tau_max),
            None => lapend::lapend_single(x, tau_min, tau_max),
        };
        z.into_pyarray(py)
    }

    Ok(())
}
