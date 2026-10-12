"""Original bounded brow/orbit-to-malar 3D control-curve study on frozen v5.

The named control chains are authored from inspection of our actual mesh and
the AI design board. They are exploratory design changes, not measured anatomy.
The cage is volumetric and its three-coordinate field may be composed with
other regional fields. No head, scan, image texture, or triangle smoothing is
used. Independent globes and the entire finite eyelid margin remain unchanged.
"""
import numpy as np
from anatomical_volume_cage import build_cage, displacement as cage_displacement

ORIGIN = np.array([-88., -64., 40.])
STEP = 8.
SHAPE = (23, 20, 15)
EYE_CENTERS = np.array([[31.3, 12., 92.1], [-31.3, 12., 92.1]])


def controls():
    records = []
    def pair(name, p, d, weight=3.):
        for sign in ([1, -1] if p[0] else [1]):
            records.append({'region': name,
                            'source': [sign*p[0], p[1], p[2]],
                            'displacement': [sign*d[0], d[1], d[2]],
                            'weight': weight})

    # Calm the pinched central block before carrying the roof into the temple.
    for p,d in [((0,28,120.2),(0,.2,-3.1)),
                ((6,29,118.8),(.15,.1,-2.0)),
                ((12,29,116.1),(.25,.1,-1.0)),
                ((18,29,114.3),(.35,.25,-.8)),
                ((24,29,112.9),(.35,.35,.6)),
                ((31,29,111.3),(.45,.35,2.2)),
                ((38,28,108.7),(.65,.4,2.8)),
                ((45,26,103.3),(.7,.35,2.9)),
                ((51,24,94.7),(.7,.3,2.2)),
                ((56,21,85.6),(.4,.1,1.0)),
                ((60,20,71.0),(0,0,0))]:
        pair('continuous_supraorbital_sweep',p,d)

    # The superior plane shares the shelf's broad lateral flow. This avoids
    # making a thin added brow bead and distributes the glabellar recession.
    for p,d in [((0,39,118.0),(0,0,-1.5)),
                ((9,39,117.0),(.1,0,-1.2)),
                ((18,38,114.5),(.2,.1,-.35)),
                ((29,37,111.2),(.35,.2,1.0)),
                ((40,36,105.4),(.45,.2,1.5)),
                ((49,34,97.0),(.4,.1,1.3)),
                ((56,32,85.0),(.2,0,.5))]:
        pair('frontal_to_brow_plane',p,d,2.5)

    # A supported roof immediately above the protected upper lid, continuous
    # into the lateral orbital wall rather than a separate medial knot.
    for p,d in [((13,22,112.8),(.15,0,1.1)),
                ((19,22,108.5),(.25,.25,1.8)),
                ((25,22,107.7),(.35,.35,2.0)),
                ((32,22,108.0),(.4,.35,1.3)),
                ((39,22,106.3),(.55,.4,1.8)),
                ((46,20,101.0),(.65,.25,2.4)),
                ((52,16,91.0),(.55,.1,1.8)),
                ((56,9,86.0),(.4,0,1.0))]:
        pair('upper_orbital_roof_and_lateral_wall',p,d,2.5)

    # Short broad fills ease the diagonal inner-orbital trough. The malar
    # crest is a continuous low roll; the lower cheek and nose are retained.
    for p,d in [((17,3,110.9),(.1,-.15,1.2)),
                ((20,-3,112.0),(.15,-.2,1.8)),
                ((23,-10,112.0),(.2,-.25,2.3)),
                ((25,-18,111.9),(.2,-.2,1.8)),
                ((27,-25,111.4),(.1,-.1,.8))]:
        pair('inner_orbit_to_malar_transition',p,d,3.)
    for p,d in [((27,-2,109.8),(.15,.15,.7)),
                ((34,-5,111.0),(.3,.25,.8)),
                ((42,-9,110.0),(.35,.35,.9)),
                ((49,-12,105.0),(.45,.35,.8)),
                ((55,-13,97.0),(.35,.25,.5)),
                ((59,-12,88.0),(.2,.1,.2))]:
        pair('lower_orbital_to_malar_roll',p,d,2.5)

    # Low-frequency retention curves, not a fixed triangle-patch boundary.
    for p in [(0,55,113),(15,54,112),(30,52,109),(46,49,99),
              (57,45,83),(63,30,60),(63,10,62),(63,-10,62),
              (60,-28,80),(48,-34,92),(36,-36,104),(23,-35,114),
              (15,-29,129),(12,-12,126),(8,0,124),(0,12,121),
              (0,18,118),(10,12,114),(12,3,117),
              (31,12,92.1),(43,12,100),(20,12,100)]:
        pair('retained_nose_forehead_and_cheek',p,(0,0,0),4.)
    return records


def _smooth(t):
    t=np.clip(t,0.,1.)
    return t*t*t*(10.-15.*t+6.*t*t)


def displacement(points, coefficients):
    """Evaluate the composable field, including exact interface protection."""
    p=np.asarray(points,dtype=float)
    out=cage_displacement(p,coefficients,protect_eye_interfaces=False,
                          origin=ORIGIN,step=STEP,shape=SHAPE)
    ax=np.abs(p[:,0]); y=p[:,1]; z=p[:,2]
    distance=np.minimum(np.linalg.norm(p-EYE_CENTERS[0],axis=1),
                        np.linalg.norm(p-EYE_CENTERS[1],axis=1))
    # All authored finite-margin loops lie inside 13.36 units. Freeze a
    # larger 14.25-unit envelope, then enter the surrounding lid plate gently.
    gate=_smooth((distance-14.25)/7.75)
    gate*=_smooth((68.-ax)/9.)*_smooth((y+43.)/13.)*_smooth((64.-y)/12.)
    gate*=_smooth((z-54.)/16.)
    # The central nasal bridge below the glabella is another exact exclusion.
    nasal_release=1.-(1.-_smooth((ax-12.)/9.))*(1.-_smooth((y-14.)/10.))
    gate*=nasal_release
    return out*gate[:,None]


def build_field():
    coefficients,report=build_cage(controls(),origin=ORIGIN,step=STEP,shape=SHAPE)
    report.update({'study':'Bounded v6 brow/orbit-to-malar plane study on frozen v5',
                   'input_sha256':'ba497c73418c1c4e31db3b0b06c07b36656f67c096fc5ce43ea399523e72ea49',
                   'rigid_globe_and_finite_margin_envelope_radius':14.25,
                   'eye_centers':EYE_CENTERS.tolist(),
                   'limits':'Visual design study; sampled Jacobians are not a global self-intersection proof.'})
    return coefficients,report


def deform(vertices,triangles,object_ids,coefficients=None):
    v=np.asarray(vertices,dtype=float).copy();f=np.asarray(triangles);ids=np.asarray(object_ids)
    if coefficients is None:coefficients,report=build_field()
    else:report={'study':'Cached bounded v6 brow/orbit-to-malar field'}
    skin=np.unique(f[ids==0]);delta=displacement(v[skin],coefficients);v[skin]+=delta
    globe=np.unique(f[ids>0])
    report.update({'maximum_vertex_displacement':float(np.linalg.norm(delta,axis=1).max()),
                   'globe_vertices_unchanged':bool(np.array_equal(v[globe],vertices[globe]))})
    return v,report,coefficients
