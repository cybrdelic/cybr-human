import assert from 'node:assert/strict';
import { regionalSkinOptics as optics } from '../src/regional_skin_optics.js';
assert.throws(() => optics(NaN, 0, 40), TypeError);
assert.equal(optics(0, -35.5, 0).lip, 0);
assert.equal(optics(26, -35, 60).lip, 0);
assert.ok(optics(0, -35.5, 60).lip > 0.99);
assert.ok(optics(0, -40, 60).lip > 0.9);
assert.ok(optics(0, -29, 60).lip < 0.01);
assert.ok(optics(0, -9, 60).roughness < optics(45, 60, 60).roughness);
for (let x = -80; x <= 80; x += 2) for (let y = -70; y <= 100; y += 2) {
  const a = optics(x, y, 45), b = optics(-x, y, 45);
  assert.deepEqual(a, b);
  assert.ok(a.roughness >= 0.3 && a.roughness <= 0.6);
  for (const field of Object.keys(a)) {
    assert.ok(Number.isFinite(a[field]));
    assert.ok(Math.abs(a[field] - optics(x + 1e-5, y, 45)[field]) < 0.001);
  }
}
console.log('Regional optics: finite bounded symmetric continuous fields and authored lip/face fixtures passed');
