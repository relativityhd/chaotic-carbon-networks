use numpy::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

fn h1nd(x_idx: ArrayView2<'_, usize>, bins: usize) -> Array1<f32> {
    let t = x_idx.shape()[0];
    let _v = x_idx.shape()[1];

    let hd = x_idx.map_axis(Axis(0), |xv_idx| {
        // Create a vector to store the histogram
        let mut hist = [0.; 128];

        // Calculate the histogram of x
        let p = 1. / t as f32;
        for &idx in xv_idx {
            hist[idx] += p;
        }

        // Calculate the normalized entropy
        let mut h = 0.;
        for i in 0..bins {
            let x = hist[i];
            if x > 0. {
                h -= x * x.ln();
            }
        }
        h
    });

    hd
}

fn h21d(x_idx: ArrayView1<'_, usize>, y_idx: ArrayView1<'_, usize>, bins: usize) -> f32 {
    let t = x_idx.shape()[0];

    // Calculate the joint entropy of x and y
    let mut flat_hist = [0.; 128 * 128];

    let p = 1. / t as f32;
    for i in 0..t {
        let idx = x_idx[i] * bins + y_idx[i];
        flat_hist[idx] += p;
    }

    // Calculate the normalized entropy
    let mut h = 0.;
    for i in 0..bins * bins {
        let x = flat_hist[i];
        if x > 0. {
            h -= x * x.ln();
        }
    }
    h
}

fn mind(x: ArrayView2<'_, f32>, y: ArrayView2<'_, f32>, bins: usize) -> Array2<f32> {
    // Expect x and y to have the same shape
    assert_eq!(x.shape(), y.shape(), "x and y must have the same shape");
    // Expect the shape of x and y to be (time [t], vertex[v])
    assert_eq!(x.ndim(), 2, "x must have 2 dimensions");
    assert_eq!(y.ndim(), 2, "y must have 2 dimensions");
    // Expect bins to be less or equal to 64
    assert!(bins <= 128, "bins must be less or equal to 128");

    let _t = x.shape()[0];
    let v = x.shape()[1];

    // Get range of x and y
    let xmin = x.iter().fold(f32::MAX, |a, &b| a.min(b));
    let xmax = x.iter().fold(f32::MIN, |a, &b| a.max(b));
    let ymin = y.iter().fold(f32::MAX, |a, &b| a.min(b));
    let ymax = y.iter().fold(f32::MIN, |a, &b| a.max(b));

    let deltax = bins as f32 / (xmax - xmin);
    let deltay = bins as f32 / (ymax - ymin);

    // Precalculate x_idx and y_idx
    let x_idx = x.mapv(|x| {
        let mut idx = ((x - xmin) * deltax) as usize;
        idx = idx.min(bins - 1);
        idx
    });

    let y_idx = y.mapv(|y| {
        let mut idx = ((y - ymin) * deltay) as usize;
        idx = idx.min(bins - 1);
        idx
    });

    // Precalculate h for x
    let hx = h1nd(x_idx.view(), bins);
    // Precalculate h for y
    let hy = h1nd(y_idx.view(), bins);

    // Precalculate slices
    //let xslices = (0..v).map(|i| x_idx.slice(s![.., i])).collect::<Vec<_>>();
    //let yslices = (0..v).map(|i| y_idx.slice(s![.., i])).collect::<Vec<_>>();

    // Calculate hxy with rayon par_iter
    let hxy = (0..v)
        .into_par_iter()
        .map(|i| {
            (0..v)
                .into_iter()
                .map(|j| h21d(x_idx.slice(s![.., i]), y_idx.slice(s![.., j]), bins))
                .collect::<Vec<f32>>()
        })
        .collect::<Vec<Vec<f32>>>();

    // Calculate the mutual information along the time axis
    let mut mi = Array2::<f32>::zeros((v, v));

    // Iterate over x vertex
    for i in 0..v {
        // Iterate over y vertex
        for j in 0..v {
            mi[[j, i]] = hx[i] + hy[j] - hxy[i][j];
        }
    }

    mi
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "rust_chaotic_carbon_networks")]
fn chaotic_carbon_networks(_py: Python, m: &PyModule) -> PyResult<()> {
    // wrapper of `nd_mi`
    #[pyfn(m)]
    #[pyo3(name = "mind")]
    fn nd_mi_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
        y: PyReadonlyArray2<'py, f32>,
        bins: usize,
    ) -> &'py PyArray2<f32> {
        let x = x.as_array();
        let y = y.as_array();
        let z = mind(x, y, bins);
        z.into_pyarray(py)
    }
    Ok(())
}
