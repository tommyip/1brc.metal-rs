fn swar_parse_temp(chars: i64, temp_len: u32) -> i16 {
    const MAGIC_MUL: i64 = 100 * 0x1000000 + 10 * 0x10000 + 1;
    let shift = 48 - (temp_len << 3);
    let sign = (!chars << 59) >> 63;
    let minus_filter = !(sign & 0xFF);
    let digits = ((chars & minus_filter) << shift) & 0x0F000F0F00;
    println!("digits {:016x}", digits);
    let abs_value = (digits.wrapping_mul(MAGIC_MUL) as u64 >> 32) & 0x3FF;
    println!("abs_value {}", abs_value);
    ((abs_value as i64 ^ sign) - sign) as i16
}

fn swar_parse_temp_short(chars: u32, temp_len: u32) -> i16 {
    const MAGIC_MUL: u64 = 100 * (1 << 32) + 10 * (1 << 24) + (1 << 8);
    let shift = 32 - (temp_len << 3);
    let digits = (chars << shift) & 0xF000F0F;
    ((digits as u64).wrapping_mul(MAGIC_MUL) as u64 >> 32) as i16 & 0x3FF
}

fn main() {
    {
        println!("original");
        let temp_str = i64::from_le_bytes(*b"42.3\n   ");
        let temp_len = 5;
        let temp = swar_parse_temp(temp_str, temp_len);
        println!("{}", temp);
    }
    {
        println!("short");
        let temp_str = u32::from_le_bytes(*b"42.3");
        let temp_len = 4;
        let temp = swar_parse_temp_short(temp_str, temp_len);
        println!("{}", temp);
        let temp_str = u32::from_le_bytes(*b"9.1\n");
        let temp_len = 3;
        let temp = swar_parse_temp_short(temp_str, temp_len);
        println!("{}", temp);
    }
}
