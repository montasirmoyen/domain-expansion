export type DomainId =
  | 'neutral'
  | 'unlimited-void'
  | 'malevolent-shrine'
  | 'self-embodiment'
  | 'idle-death-gamble'
  | 'authentic-mutual-love'
  | 'yuji-domain'

export type Landmark = { x: number; y: number; z: number }

export const DOMAIN_META: Record<DomainId, { name: string; japanese: string; owner: string; accent: string; description: string }> = {
  neutral: { name: 'Cursed Energy', japanese: '呪力', owner: 'Awaiting manifestation', accent: '#75e9e5', description: 'Raise your hands and form a domain expansion sign.' },
  'unlimited-void': { name: 'Unlimited Void', japanese: '無量空処', owner: 'Satoru Gojo', accent: '#88e5ff', description: 'An endless cosmos where boundless information floods the senses.' },
  'malevolent-shrine': { name: 'Malevolent Shrine', japanese: '伏魔御廚子', owner: 'Ryomen Sukuna', accent: '#ff3b35', description: 'A barrierless, demonic shrine that cleaves everything within its reach.' },
  'self-embodiment': { name: 'Self-Embodiment of Perfection', japanese: '自閉円頓裹', owner: 'Mahito', accent: '#c66bff', description: 'A black flower of clasped hands where every soul rests in the caster’s palm.' },
  'idle-death-gamble': { name: 'Idle Death Gamble', japanese: '坐殺博徒', owner: 'Kinji Hakari', accent: '#ffd43b', description: 'A feverish pachinko game of shutters, numbers, trains, and a 1-in-239 jackpot.' },
  'authentic-mutual-love': { name: 'Authentic Mutual Love', japanese: '真贋相愛', owner: 'Yuta Okkotsu', accent: '#ff9cd6', description: 'Cross-shaped wreckage and endless katanas, each holding a copied technique.' },
  'yuji-domain': { name: 'Unnamed Domain', japanese: '領域展開', owner: 'Yuji Itadori', accent: '#ff765c', description: 'A quiet reconstruction of Yuji’s hometown: station, streets, and treasured memories.' },
}
