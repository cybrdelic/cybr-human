import fs from 'node:fs';import assert from 'node:assert/strict';
import {certifyMovingTrianglePair as certify} from '../src/exact_moving_triangle_pair.mjs';
import {verifyTriangleIntersectionFeature as feature} from '../../../procedural/mechanics/regional-reference/workspace/human-mechanics-20261010/src/prototypes/exact_triangle_contact_feature.mjs';
const fixture=JSON.parse(fs.readFileSync(new URL('./actual-rejected-state-fixture.json',import.meta.url))),results=[];
for(const q of fixture.pairs){
 assert(feature(...q.rest).valid);assert(feature(...q.valid08).valid);
 assert(!feature(...q.invalid16).valid);
 const rest=certify(q.rest[0],q.rest[0],q.rest[1],q.rest[1]),valid08=certify(q.valid08[0],q.valid08[0],q.valid08[1],q.valid08[1]),invalid16=certify(q.invalid16[0],q.invalid16[0],q.invalid16[1],q.invalid16[1]),toward16=certify(q.valid08[0],q.invalid16[0],q.valid08[1],q.invalid16[1]);
 assert(rest.certified);assert(valid08.certified);assert(!invalid16.certified);assert(!toward16.certified);
 results.push({faces:q.original_boundary_face_ids,rest,valid08,invalid16,toward16});
}
console.log(JSON.stringify({actualPairs:2,exactFeatureChecksPassed:true,invalidEndpointAndInterpolatedPathRejected:true,results},null,2));
