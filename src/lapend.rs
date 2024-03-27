use std::usize;

use ndarray::Zip;
use ndarray::{s, Array2, ArrayView2};
use rayon::prelude::*;

// TODO: Use this: https://docs.rs/ndarray/latest/ndarray/parallel/index.html

pub fn lapend_single(x: ArrayView2<'_, f32>, tau_min: isize, tau_max: isize) -> Array2<f32> {
    // Expect the shape of x to be (time [t], vertex[v])
    assert_eq!(x.ndim(), 2, "x must have 2 dimensions");

    let t = x.shape()[0] as isize;
    let v = x.shape()[1];

    let taud = (tau_max - tau_min) as usize;
    assert!(tau_min > 0, "tau_min must be larger than 0");
    assert!(tau_max > tau_min, "tau_max must be larger than tau_min");
    assert!(t > (tau_max + 2), "tau_max + 2 must be smaller than t");

    // Precalc std and means for every ttau

    let mut meanx = Array2::zeros((v, taud));
    let mut stdx = Array2::zeros((v, taud));

    // Shift y by tau and clip x
    for (ti, tau) in (tau_min..tau_max).enumerate() {
        let nt = t - tau;

        for i in 0..v {
            let xtau = x.slice(s![..-tau, i]);
            let mx = xtau.mean().unwrap();
            meanx[[i, ti]] = mx;
            stdx[[i, ti]] = (xtau.mapv(|x| (x - mx).powi(2)).sum() / nt as f32).sqrt();
        }
    }

    let mut rho = Array2::zeros((v, v));

    rho.indexed_iter_mut()
        .par_bridge()
        .for_each(|((i, j), val)| {
            let mut max_rhot = 0.;
            for (ti, tau) in (tau_min..tau_max).enumerate() {
                let nt = t - tau;

                let meanxti = meanx[[i, ti]];
                let meanyti = meanx[[j, ti]];

                let xti = x.slice(s![..-(tau), i]);
                let yti = x.slice(s![tau.., j]);
                let cov = Zip::from(xti)
                    .and(yti)
                    .fold(0., |acc, a, b| acc + (a - meanxti) * (b - meanyti))
                    / nt as f32;

                let corr = (cov / (stdx[[i, ti]] * stdx[[j, ti]])).abs();
                if corr > max_rhot {
                    max_rhot = corr;
                }
            }
            *val = max_rhot;
        });

    rho
}

pub fn lapend_double(
    x: ArrayView2<'_, f32>,
    y: ArrayView2<'_, f32>,
    tau_min: isize,
    tau_max: isize,
) -> Array2<f32> {
    // Expect the shape of x and y to be (time [t], vertex[v])
    assert_eq!(x.ndim(), 2, "x must have 2 dimensions");
    assert_eq!(y.ndim(), 2, "y must have 2 dimensions");

    let t = x.shape()[0] as isize;
    let ty = y.shape()[0] as isize;
    let vx = x.shape()[1];
    let vy = y.shape()[1];
    assert_eq!(t, ty, "x and y must have same t-dimension");

    let taud = (tau_max - tau_min) as usize;
    assert!(tau_min > 0, "tau_min must be larger than 0");
    assert!(tau_max > tau_min, "tau_max must be larger than tau_min");
    assert!(t > (tau_max + 2), "tau_max + 2 must be smaller than t");

    // Precalc std and means for every ttau

    let mut meanx = Array2::zeros((vx, taud));
    let mut meany = Array2::zeros((vy, taud));
    let mut stdx = Array2::zeros((vx, taud));
    let mut stdy = Array2::zeros((vy, taud));

    // Shift y by tau and clip x
    for (ti, tau) in (tau_min..tau_max).enumerate() {
        let nt = t - tau;

        for i in 0..vx {
            let xtau = x.slice(s![..-tau, i]);
            let mx = xtau.mean().unwrap();
            meanx[[i, ti]] = mx;
            stdx[[i, ti]] = (xtau.mapv(|x| (x - mx).powi(2)).sum() / nt as f32).sqrt();
        }

        for j in 0..vy {
            let ytau = y.slice(s![tau.., j]);
            let my = ytau.mean().unwrap();
            meany[[j, ti]] = my;
            stdy[[j, ti]] = (ytau.mapv(|y| (y - my).powi(2)).sum() / nt as f32).sqrt();
        }
    }

    let mut rho = Array2::zeros((vx, vy));

    rho.indexed_iter_mut()
        .par_bridge()
        .for_each(|((i, j), val)| {
            let mut max_rhot = 0.;
            for (ti, tau) in (tau_min..tau_max).enumerate() {
                let nt = t - tau;

                let meanxti = meanx[[i, ti]];
                let meanyti = meany[[j, ti]];

                let xti = x.slice(s![..-(tau), i]);
                let yti = y.slice(s![tau.., j]);
                let cov = Zip::from(xti)
                    .and(yti)
                    .fold(0., |acc, a, b| acc + (a - meanxti) * (b - meanyti))
                    / nt as f32;

                let corr = (cov / (stdx[[i, ti]] * stdy[[j, ti]])).abs();
                if corr > max_rhot {
                    max_rhot = corr;
                }
            }
            *val = max_rhot;
        });

    rho
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::lapend_double;

    #[test]
    fn it_works() {
        let x = Array2::zeros((48, 20)) + 1.;
        let y = Array2::zeros((48, 20)) + 1.;
        let z = lapend_double(x.view(), y.view(), 2, 20);
        println!("{:?}", z.sum());
    }
}
