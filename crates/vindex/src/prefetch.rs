//! Asking the chip to start loading memory before it is needed, so several reads overlap instead of waiting
//! one after another. Hints only: results never change.

/// The chip's cache line: 128 bytes on Apple's M chips (`sysctl hw.cachelinesize`), 64 on most others.
#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
pub const LINE: usize = 128;
#[cfg(not(all(target_arch = "aarch64", target_vendor = "apple")))]
pub const LINE: usize = 64;
/// Regions up to this long get a hint every 64 bytes; longer ones one hint per line. Measured on the M3 Max (step
/// 20): one hint per 128-byte line made 768-byte rows 14% faster (half the hints, less queueing) but 200-byte rows
/// 5% slower.
const SHORT: usize = 256;

/// Start loading the memory that holds `bytes` bytes from `ptr`.
#[inline(always)]
pub fn prefetch_bytes(ptr: *const u8, bytes: usize) {
    if bytes <= SHORT {
        let mut off = 0;
        while off < bytes {
            prefetch_line(ptr.wrapping_add(off));
            off += 64;
        }
        return;
    }
    // one hint per line, from the line holding the first byte to the line holding the last (rows rarely start on a
    // line boundary)
    let lead = ptr as usize & (LINE - 1);
    let first = ptr.wrapping_sub(lead);
    let mut off = 0;
    while off < lead + bytes {
        prefetch_line(first.wrapping_add(off));
        off += LINE;
    }
}

#[inline(always)]
fn prefetch_line(p: *const u8) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: PRFM is a hint; it never faults, even on an invalid address, and touches no registers or flags.
    unsafe {
        std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, readonly, preserves_flags));
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: PREFETCHT0 is a hint and never faults.
    unsafe {
        std::arch::x86_64::_mm_prefetch(p as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = p;
}
