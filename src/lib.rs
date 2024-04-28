/// https://stackoverflow.com/a/28124775
pub fn round_to_positive(x: f32) -> f32 {
    let y = x.floor();
    if x == y {
        x
    } else {
        let z = (2.0 * x - y).floor();
        z.copysign(x)
    }
}
