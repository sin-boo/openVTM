const ADJECTIVES = [
  'amber',
  'brisk',
  'coral',
  'delta',
  'ember',
  'frost',
  'gleam',
  'harbor',
  'ivory',
  'jade',
  'keen',
  'lunar',
  'mist',
  'nova',
  'onyx',
  'plume',
  'quartz',
  'rapid',
  'solar',
  'tide',
  'ultra',
  'vivid',
  'wave',
  'zenith',
]

const NOUNS = [
  'otter',
  'falcon',
  'cedar',
  'pixel',
  'ridge',
  'comet',
  'orchid',
  'anvil',
  'beacon',
  'cinder',
  'drizzle',
  'echo',
  'flint',
  'grove',
  'horizon',
  'iris',
  'jasper',
  'kite',
  'lagoon',
  'maple',
  'nectar',
  'orbit',
  'pebble',
  'quill',
]

export function randomServerName(): string {
  const a = ADJECTIVES[Math.floor(Math.random() * ADJECTIVES.length)]!
  const n = NOUNS[Math.floor(Math.random() * NOUNS.length)]!
  const suffix = Math.floor(Math.random() * 90 + 10)
  return `${a}-${n}-${suffix}`
}
