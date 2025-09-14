#![doc = include_str!("../README.md")]
#![no_std]

use core::{
    cmp::min, fmt::{Debug, UpperHex}, ops::{Bound, RangeBounds}
};

use num_traits::{CheckedShr, PrimInt, Unsigned};

#[inline]
fn bitmask<T: PrimInt + Unsigned>(bit_width: usize) -> T {
    let max_width = core::mem::size_of::<T>() * 8;
    assert!(bit_width <= max_width);
    if bit_width == max_width {
        T::max_value()
    } else {
        T::from((1_usize << bit_width) - 1).unwrap()
    }
}

// The "rng" helper here is renamed but based on Rust lang forum by "cuviper":
// https://internals.rust-lang.org/t/the-need-for-decreasing-range-syntax-5-0/15742/7
fn rng(a:usize, b: usize) -> impl Iterator<Item=usize> {
    // Helper for creating range-like iterators counting either up _or_ down without type issues
    let (part1, part2) = if a <= b {
        (a..=b, 1..=0)
    } else {
        (1..=0, b..=a)
    };
    part1.chain(part2.rev())
}

impl<T> BitfieldAccess for T where T: AsRef<[u8]> {}


pub trait BitfieldAccess: AsRef<[u8]> {
    /// Read a bitfield with the given bit indices from a buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use bitfield_access::BitfieldAccess;
    ///
    /// let buffer = [0x12, 0x34, 0x56, 0x78];
    /// assert_eq!(buffer.read_field::<u8>(4..8), 0x2);
    /// assert_eq!(buffer.read_field::<u16>(12..24), 0x456);
    /// assert_eq!(buffer.read_field::<u8>(25..=25), 0x1);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the range of bits is wider than the integer type `T`
    /// or the bit indices are out of bounds.
    #[inline]
    fn _read_field<T>(&self, bitrange: impl RangeBounds<usize>, is_msb0: bool) -> T
    where
        T: PrimInt + Unsigned,
    {
        // There's a lot of logic here, but as an inline function the bit range is
        // typically known at compile time, reducing this to just a small handful
        // of shifts and bitwise instructions.
        let data = self.as_ref();
        let start = match bitrange.start_bound() {
            core::ops::Bound::Included(idx) => *idx,
            core::ops::Bound::Excluded(idx) => *idx + 1,
            core::ops::Bound::Unbounded => 0,
        };
        let end = match bitrange.end_bound() {
            core::ops::Bound::Included(idx) => *idx + 1,
            core::ops::Bound::Excluded(idx) => *idx,
            core::ops::Bound::Unbounded => data.len() * 8,
        };

        let storage_width = 8 * core::mem::size_of::<T>();
        let bit_width = end - start;
        assert!(
            bit_width <= storage_width,
            "field width {} exceeds storage width {}",
            bit_width,
            storage_width
        );
        let first_byte = start / 8;
        let last_byte = (end - 1) / 8;
        let offset = match is_msb0 {
            true => 7 - (end - 1) % 8,
            false => start % 8,
        };
        let mask = bitmask(bit_width);

        // Build the result depending on bit numbering
        let mut byte_orderer = match is_msb0 {
            true => rng(last_byte, first_byte),
            false => rng(first_byte, last_byte),
        };

        let i = byte_orderer.next().unwrap();
        let mut result =  match is_msb0 {
            true => T::from(data[i]).unwrap() >> offset,  // First possibly partial byte, shifted down to LSBit
            false => (T::from(data[i]).unwrap() >> offset) & bitmask(min(bit_width, 8-offset)), // Shift and mask
        };
        for (n, i ) in byte_orderer.enumerate() {
            result = result | T::from(data[i]).unwrap() << (8 * (n+1) - offset);  // Following bytes, shifted "just in front" of res
        }
        result & mask  // Finally clipping to just the wanted bits
    }

    #[inline]
    fn read_field<T>(&self, bitrange: impl RangeBounds<usize>) -> T
    where
        T: PrimInt + Unsigned,
    {
        return self._read_field(bitrange, true);
    }

    #[inline]
    fn read_field_lsb0<T>(&self, bitrange: impl RangeBounds<usize>) -> T
    where
        T: PrimInt + Unsigned,
    {
        return self._read_field(bitrange, false);
    }


    /// Write a bitfield with the given bit indices to a buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use bitfield_access::BitfieldAccess;
    ///
    /// let mut buffer = [0x12, 0x34, 0x56, 0x78];
    /// buffer.write_field(4..8, 0xA_u8);
    /// assert_eq!(buffer, [0x1A, 0x34, 0x56, 0x78]);
    /// buffer.write_field(20..=27, 0xBC_u8);
    /// assert_eq!(buffer, [0x1A, 0x34, 0x5B, 0xC8]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the bit indices are out of bounds or the value is too large.


    #[inline]
    fn _write_field<T>(&mut self, bitrange: impl RangeBounds<usize>, mut value: T, is_msb0: bool)
    where
        Self: AsMut<[u8]>,
        T: PrimInt + Unsigned + TryInto<u8> + UpperHex + CheckedShr,
        <T as TryInto<u8>>::Error: Debug,
    {
        // There's a lot of logic here, but as an inline function the bit range is
        // typically known at compile time, reducing this to just a small handful
        // of shifts and bitwise instructions.
        let data = self.as_mut();
        let mut start = match bitrange.start_bound() {
            Bound::Included(idx) => *idx,
            Bound::Excluded(idx) => *idx + 1,
            Bound::Unbounded => 0,
        };
        let mut end = match bitrange.end_bound() {
            Bound::Included(idx) => *idx + 1,
            Bound::Excluded(idx) => *idx,
            Bound::Unbounded => data.len() * 8,
        };
        let first_byte = start / 8;
        let last_byte = (end - 1) / 8;
        let max_value = bitmask(end - start);
        assert!(
            value <= max_value,
            "value {:#X} exceeds maximum field value {:#X}",
            value,
            max_value
        );

        let byte_mask = T::from(0xFF).unwrap();
        let zero = T::from(0x0).unwrap();

        // write in one-byte chunks, from the last (LSB) to the first
        let byte_orderer = match is_msb0 {
            true => rng(last_byte, first_byte),
            false => rng(first_byte, last_byte),
        };

        for i in byte_orderer {
            let bit_offset = match is_msb0 {
                true => 7 - (end - 1) % 8,
                false => start % 8,
            };
            let bit_width = core::cmp::min(8 - bit_offset, end - start);
            let bit_mask = bitmask::<u8>(bit_width) << bit_offset;
            let new_bits: u8 = (value & byte_mask).try_into().unwrap();
            data[i] = (data[i] & !bit_mask) | ((new_bits << bit_offset) & bit_mask);
            match is_msb0 {
                true => {
                    end -= bit_width;
                },
                false => {
                    start += bit_width;
                },
            }
            value = value.checked_shr(bit_width as u32).unwrap_or(zero);
        }
    }

    #[inline]
    fn write_field<T>(&mut self, bitrange: impl RangeBounds<usize>, value: T)
    where
        Self: AsMut<[u8]>,
        T: PrimInt + Unsigned + TryInto<u8> + UpperHex + CheckedShr,
        <T as TryInto<u8>>::Error: Debug,
    {
        return self._write_field(bitrange, value, true);
    }


    #[inline]
    fn write_field_lsb0<T>(&mut self, bitrange: impl RangeBounds<usize>, value: T)
    where
        Self: AsMut<[u8]>,
        T: PrimInt + Unsigned + TryInto<u8> + UpperHex + CheckedShr,
        <T as TryInto<u8>>::Error: Debug,
    {
        return self._write_field(bitrange, value, false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[inline(never)]
    fn test_rng_helper() {
        assert_eq!(rng(1, 2).next(), Some(1));  // Counting up
        assert_eq!(rng(2, 1).next(), Some(2));  // Counting down
    }

    #[test]
    #[inline(never)]
    fn test_read_field() {
        let single_byte = [0b10101111u8];
        let buffer = [0x12, 0x34, 0x56, 0x78];
        let acc = [55u8, 124, 248, 210, 0x7c, 0x18, 0xf8]; // From the field, bits 36..52 = 0x8187

        // Test difference in LSB0 and MSB0 bit ordering
        assert_eq!(single_byte.read_field::<u8>(0..4), 0b1010u8);
        assert_eq!(single_byte.read_field_lsb0::<u8>(0..4), 0b1111u8);
        assert_eq!(single_byte.read_field_lsb0::<u8>(1..=5), 0b10111);  // Get cross-nibble bits
        assert_eq!(single_byte.read_field::<u8>(3..=5), 0b011);   // Same bits, different adressing

        // Test reading a single byte
        assert_eq!(buffer.read_field::<u8>(4..8), 0x2);
        assert_eq!(buffer.read_field::<u8>(8..16), 0x34);

        // From the field
        assert_eq!(acc.read_field_lsb0::<u16>(36..52), 0x8187);

        // Test reading across byte boundaries
        assert_eq!(buffer.read_field::<u16>(4..20), 0x2345);
        assert_eq!(buffer.read_field_lsb0::<u16>(4..20), 0x6341);
        assert_eq!(buffer.read_field_lsb0::<u8>(20..28), 0x85);
        assert_eq!(buffer.read_field_lsb0::<u8>(6..=10), 0b10000u8);

        // Test reading the entire buffer
        assert_eq!(buffer.read_field::<u32>(..), 0x12345678);
        assert_eq!(buffer.read_field_lsb0::<u32>(..), 0x78563412);

        // Test reading a single bit
        assert_eq!(buffer.read_field::<u8>(7..8), 0x0);
        assert_eq!(buffer.read_field::<u8>(17..=17), 0x1);
    }

    #[test]
    fn test_write_field() {
        const BUFFER: [u8; 4] = [0x12, 0x34, 0x56, 0x78];

        // Test writing to a single-byte buffer
        let mut buffer = [0];
        buffer.write_field::<u8>(1..=3, 0x3);
        assert_eq!(buffer, [0b00110000u8]);
        buffer[0] = 0;  // Clear to all-zeroes
        buffer.write_field_lsb0::<u8>(1..=3, 0x3);  // Same value and "symbolic" address, different result
        assert_eq!(buffer, [0b00000110u8]);

        // Test writing a single byte
        let mut buffer = BUFFER;
        buffer.write_field::<u8>(4..8, 0xA);
        assert_eq!(buffer, [0x1A, 0x34, 0x56, 0x78]);
        buffer.write_field::<u8>(0..8, 0xBC);
        assert_eq!(buffer, [0xBC, 0x34, 0x56, 0x78]);
        buffer.write_field::<u8>(28..32, 0x2);
        assert_eq!(buffer, [0xBC, 0x34, 0x56, 0x72]);
        buffer.write_field::<u8>(28..32, 0x8);  // Restore...
        assert_eq!(buffer, [0xBC, 0x34, 0x56, 0x78]);

        // Test writing across byte boundaries
        let mut buffer = BUFFER;
        buffer.write_field::<u8>(12..20, 0xBC);
        assert_eq!(buffer, [0x12, 0x3B, 0xC6, 0x78]);
        //let mut buffer = [0u8; 4];
        buffer.write_field_lsb0::<u8>(20..28, 0xDE);
        assert_eq!(buffer, [0x12, 0x3B, 0xE6, 0x7D]);

        // Test writing the entire buffer
        let mut buffer = BUFFER;
        buffer.write_field::<u32>(.., 0x87654321u32);
        assert_eq!(buffer, [0x87, 0x65, 0x43, 0x21]);

        // Test writing a single bit
        let mut buffer = BUFFER;
        buffer.write_field::<u8>(7..8, 0x1);
        assert_eq!(buffer, [0x13, 0x34, 0x56, 0x78]);
        buffer.write_field::<u8>(8..=8, 0x1);
        assert_eq!(buffer, [0x13, 0xB4, 0x56, 0x78]);
        buffer.write_field::<u8>(30..31, 0x1);
        assert_eq!(buffer, [0x13, 0xB4, 0x56, 0x7A]);
    }
}
