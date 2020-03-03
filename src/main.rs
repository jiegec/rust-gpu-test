use accel::*;
use accel_derive::kernel;
use anyhow::Result;

#[kernel]
#[dependencies("accel-core" = "0.3.0-alpha.1")]
pub unsafe fn add(a: *const f64, b: *const f64, c: *mut f64, n: usize) {
    let i = accel_core::index();
    if (i as usize) < n {
        *c.offset(i) = *a.offset(i) + *b.offset(i);
    }
}

fn main() -> Result<()> {
    let n = 32;
    let mut a = UVec::new(n)?;
    let mut b = UVec::new(n)?;
    let mut c = UVec::new(n)?;

    for i in 0..n {
        a[i] = i as f64;
        b[i] = 2.0 * i as f64;
    }
    println!("a = {:?}", a.as_slice());
    println!("b = {:?}", b.as_slice());

    let device = driver::Device::nth(0)?;
    let ctx = device.create_context_auto()?;
    let grid = Grid::x(1);
    let block = Block::x(n as u32);
    add(&ctx, grid, block, a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), n).expect("Kernel call failed");

    device::sync()?;
    println!("c = {:?}", c.as_slice());

    Ok(())
}
