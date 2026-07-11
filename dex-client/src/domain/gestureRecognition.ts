import type { DomainId, Landmark } from './domainTypes'

const d3 = (a: Landmark, b: Landmark) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z)
const angle = (a: Landmark, b: Landmark, c: Landmark) => {
  const ab = [a.x - b.x, a.y - b.y, a.z - b.z]
  const cb = [c.x - b.x, c.y - b.y, c.z - b.z]
  const dot = ab[0] * cb[0] + ab[1] * cb[1] + ab[2] * cb[2]
  const denom = Math.hypot(...ab) * Math.hypot(...cb)
  return denom ? Math.acos(Math.max(-1, Math.min(1, dot / denom))) * 180 / Math.PI : 0
}
const extended = (h: Landmark[], mcp: number, pip: number, tip: number, threshold = 155) => angle(h[mcp], h[pip], h[tip]) >= threshold
const curled = (h: Landmark[], mcp: number, pip: number, tip: number, threshold = 130) => angle(h[mcp], h[pip], h[tip]) <= threshold
const scale = (h: Landmark[]) => Math.max(d3(h[0], h[9]), 1e-6)
const fist = (h: Landmark[]) => [[5,6,8],[9,10,12],[13,14,16],[17,18,20]].every(([a,b,c]) => curled(h,a,b,c,135))
const openThumbIn = (h: Landmark[]) => {
  const straight = [[5,6,8],[9,10,12],[13,14,16],[17,18,20]].filter(([a,b,c]) => extended(h,a,b,c)).length
  return straight >= 3 && d3(h[4], h[9]) / scale(h) < .55
}
const pairs = (hands: Landmark[][], matcher: (a: Landmark[], b: Landmark[]) => boolean) => {
  for (let i = 0; i < hands.length - 1; i++) for (let j = i + 1; j < hands.length; j++) if (matcher(hands[i], hands[j])) return true
  return false
}

const yuta = (a: Landmark[], b: Landmark[]) => {
  const s = (scale(a) + scale(b)) / 2
  return ((fist(a) && openThumbIn(b)) || (fist(b) && openThumbIn(a))) && d3(a[0], b[0]) / s <= 1.8
}
const hakari = (a: Landmark[], b: Landmark[]) => {
  const [upper, lower] = a[0].y < b[0].y ? [a,b] : [b,a]
  const s = (scale(upper) + scale(lower)) / 2
  const upperOk = d3(upper[4], upper[8]) / s < .35 && [[9,10,12],[13,14,16],[17,18,20]].every(([x,y,z]) => extended(upper,x,y,z))
  const lowerOk = [[5,6,8],[9,10,12],[13,14,16],[17,18,20]].every(([x,y,z]) => extended(lower,x,y,z))
  const gap = lower[9].y - upper[0].y
  return upperOk && lowerOk && gap > .1 && gap < 1
}
const yuji = (a: Landmark[], b: Landmark[]) => {
  const s = (scale(a) + scale(b)) / 2
  const vertical = (h: Landmark[]) => Math.abs(h[8].y - h[5].y) > Math.abs(h[8].x - h[5].x) * 1.8
  const wrists = d3(a[0],b[0]) / s
  return extended(a,5,6,8) && extended(b,5,6,8) && vertical(a) && vertical(b) && d3(a[8],b[8]) / s <= .4 && wrists >= .3 && wrists <= 1.5 && (a[8].y-a[5].y)*(b[8].y-b[5].y) >= 0
}
const mahito = (a: Landmark[], b: Landmark[]) => {
  const s = (scale(a) + scale(b)) / 2
  const center = (h: Landmark[]) => [0,5,9,13,17].reduce((p,i) => [p[0]+h[i].x/5,p[1]+h[i].y/5],[0,0])
  const [ca,cb] = [center(a),center(b)]
  const palmGap = Math.hypot(ca[0]-cb[0],ca[1]-cb[1]) / s
  const wrist = d3(a[0],b[0]) / s
  return d3(a[20],b[20])/s <= .9 && d3(a[4],b[4])/s <= 1 && palmGap >= .6 && palmGap <= 2.2 && wrist >= .4 && wrist <= 3.5
}
const sukuna = (a: Landmark[], b: Landmark[]) => {
  const s = (scale(a)+scale(b))/2
  const touches = [12,16,20].filter((i) => d3(a[i],b[i])/s <= .75).length
  const left = d3(a[12],a[16])/s, right = d3(b[12],b[16])/s, wrists=d3(a[0],b[0])/s
  return touches >= 2 && left >= .25 && left <= 1.8 && right >= .25 && right <= 1.8 && wrists >= .4 && wrists <= 4.2
}
const gojo = (h: Landmark[]) => extended(h,5,6,8) && extended(h,9,10,12) && curled(h,13,14,16) && curled(h,17,18,20) && d3(h[8],h[12])/scale(h) <= .65

export const detectDomain = (hands: Landmark[][]): DomainId | null => {
  if (hands.length === 2 && yuta(hands[0],hands[1])) return 'authentic-mutual-love'
  if (hands.length === 2 && hakari(hands[0],hands[1])) return 'idle-death-gamble'
  if (hands.length >= 2 && sukuna(hands[0],hands[1])) return 'malevolent-shrine'
  if (hands.length >= 2 && pairs(hands,yuji)) return 'yuji-domain'
  if (hands.length >= 2 && pairs(hands,mahito)) return 'self-embodiment'
  if (hands.length === 1 && gojo(hands[0])) return 'unlimited-void'
  return null
}

export class PredictionSmoother {
  private history: (DomainId | null)[] = []
  push(value: DomainId | null) {
    this.history.push(value)
    if (this.history.length > 8) this.history.shift()
    const counts = new Map<DomainId,number>()
    this.history.forEach((v) => v && counts.set(v,(counts.get(v) ?? 0)+1))
    let winner: DomainId | null = null, votes = 0
    counts.forEach((count,id) => { if (count > votes) { winner=id; votes=count } })
    return votes >= 4 && votes >= Math.floor(this.history.length/2)+1 ? winner : null
  }
  reset(){ this.history=[] }
}
