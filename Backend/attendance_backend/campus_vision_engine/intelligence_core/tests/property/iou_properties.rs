// ── iou tests ─────────────────────────────────────────────────────────────────

use intelligence_core::utils::{Rect, iou};

#[test]
fn iou_no_overlap_is_zero() {
    let a = Rect { x: 0,   y: 0, width: 10, height: 10 };
    let b = Rect { x: 20, y: 20, width: 10, height: 10 };
    assert_eq!(iou(&a, &b), 0.0);
}

#[test]
fn iou_identical_rects_is_one() {
    let r = Rect { x: 5, y: 5, width: 20, height: 20 };
    let result = iou(&r, &r);
    assert!(
        (result - 1.0).abs() < 1e-6,
        "identical rects iou: {}", result
    );
}

#[test]
fn iou_partial_overlap_correct() {
    // a: [0..10] x [0..10] = area 100
    // b: [5..15] x [5..15] = area 100
    // intersection: [5..10] x [5..10] = 25
    // union: 200 - 25 = 175
    // iou = 25/175 ≈ 0.1428
    let a = Rect { x: 0, y: 0, width: 10, height: 10 };
    let b = Rect { x: 5, y: 5, width: 10, height: 10 };
    let result = iou(&a, &b);
    let expected = 25.0 / 175.0;
    assert!(
        (result - expected).abs() < 1e-5,
        "partial overlap: got {} expected {}", result, expected
    );
}

#[test]
fn iou_touching_edges_is_zero() {
    // share edge but no area
    let a = Rect { x: 0, y: 0, width: 10, height: 10 };
    let b = Rect { x: 10, y: 0, width: 10, height: 10 };
    assert_eq!(iou(&a, &b), 0.0);
}