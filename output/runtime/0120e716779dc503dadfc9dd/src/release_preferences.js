/** Public choices; numerical and device capability gates remain in the solver. */
export function releasePreferences(search = '') {
  const query = new URLSearchParams(search);
  return {
    regional: !['classic', 'default'].includes(query.get('appearance')),
    refractiveEyes: !['specular', 'legacy'].includes(query.get('eyeOptics')),
    coverageSamples: ['single', '0'].includes(query.get('surfaceCoverage')) ? 0 : 4,
    bridge: query.get('bridge') === 'versioned' ? 'versioned' : 'pipeline',
  };
}
export function releaseChoiceURL(href, kind, value) {
  const url = new URL(href);
  if (kind === 'appearance') {
    const refined = value === 'refined';
    url.searchParams.set('appearance', refined ? 'regional' : 'classic');
    url.searchParams.set('eyeOptics', refined ? 'refractive' : 'specular');
    url.searchParams.set('surfaceCoverage', refined ? 'msaa' : 'single');
  } else if (kind === 'simulation') {
    url.searchParams.set('bridge', value === 'serial' ? 'versioned' : 'pipeline');
  } else throw Error('Unknown face choice');
  return url.href;
}
