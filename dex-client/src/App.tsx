import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import { DomainCanvas } from './visuals/DomainCanvas'
import { DOMAIN_META, type DomainId } from './domain/domainTypes'
import { useHandTracking } from './vision/useHandTracking'

const DEMO_ORDER: DomainId[] = [
  'unlimited-void',
  'malevolent-shrine',
  'self-embodiment',
  'idle-death-gamble',
  'authentic-mutual-love',
  'yuji-domain',
]

function App() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const overlayRef = useRef<HTMLCanvasElement>(null)
  const [activeDomain, setActiveDomain] = useState<DomainId>('neutral')
  const [previousDomain, setPreviousDomain] = useState<DomainId>('neutral')
  const [transitionKey, setTransitionKey] = useState(0)
  const [cameraEnabled, setCameraEnabled] = useState(false)
  const [demoIndex, setDemoIndex] = useState(-1)

  const { status, prediction, handCount, enableCamera } = useHandTracking({
    videoRef,
    overlayRef,
    enabled: cameraEnabled,
  })

  useEffect(() => {
    if (demoIndex >= 0 && !prediction) return
    const nextDomain = prediction ?? 'neutral'
    if (nextDomain === activeDomain) return
    // Recognition is an external camera event; commit it to the presentation state.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (demoIndex >= 0) setDemoIndex(-1)
    setPreviousDomain(activeDomain)
    setActiveDomain(nextDomain)
    setTransitionKey((key) => key + 1)
  }, [prediction, activeDomain, demoIndex])

  const meta = DOMAIN_META[activeDomain]
  const statusText = useMemo(() => {
    if (demoIndex >= 0) return 'Preview mode · click again to explore'
    if (status === 'ready') return handCount ? `Tracking ${handCount} hand${handCount > 1 ? 's' : ''}` : 'Show a domain hand sign'
    if (status === 'loading') return 'Awakening the hand tracker…'
    if (status === 'denied') return 'Camera permission was denied'
    if (status === 'error') return 'Hand tracking could not start'
    return 'Camera is off · preview the worlds below'
  }, [demoIndex, handCount, status])

  const startCamera = async () => {
    setDemoIndex(-1)
    setCameraEnabled(true)
    await enableCamera()
  }

  const previewNext = () => {
    const next = (demoIndex + 1) % DEMO_ORDER.length
    setPreviousDomain(activeDomain)
    setActiveDomain(DEMO_ORDER[next])
    setDemoIndex(next)
    setTransitionKey((key) => key + 1)
  }

  return (
    <main className="experience" style={{ '--domain-accent': meta.accent } as React.CSSProperties}>
      <DomainCanvas active={activeDomain} previous={previousDomain} transitionKey={transitionKey} />
      <div className="cinematic-bars" aria-hidden="true" />
      <div className="grain" aria-hidden="true" />

      <header className="masthead">
        <div className="brand-mark">領域展開</div>
        <div className="brand-copy">
          <span>DOMAIN</span>
          <span>EXPANSION</span>
        </div>
      </header>

      <section className="domain-title" key={`${activeDomain}-${transitionKey}`}>
        <p className="eyebrow">{meta.owner}</p>
        <h1>{meta.name}</h1>
        <p className="japanese">{meta.japanese}</p>
        <div className="title-rule" />
        <p className="description">{meta.description}</p>
      </section>

      <aside className={`camera-card ${cameraEnabled ? 'is-live' : ''}`}>
        <div className="camera-header">
          <span className={`live-dot ${status === 'ready' ? 'active' : ''}`} />
          <span>{statusText}</span>
          <span className="hand-count">{handCount ? `0${handCount}` : '—'}</span>
        </div>
        <div className="camera-viewport">
          <video ref={videoRef} muted playsInline />
          <canvas ref={overlayRef} />
          {!cameraEnabled && (
            <button className="camera-start" type="button" onClick={startCamera}>
              <span>Enable webcam</span>
              <small>Video stays on this device</small>
            </button>
          )}
          {cameraEnabled && status === 'denied' && (
            <div className="camera-error">Allow camera access in your browser, then reload.</div>
          )}
        </div>
        <div className="camera-footer">
          <span>MEDIAPIPE / LOCAL</span>
          <span>GESTURE LOCK {prediction ? 'ON' : '—'}</span>
        </div>
      </aside>

      <nav className="controls" aria-label="Experience controls">
        <button type="button" onClick={previewNext}>Preview domains <span>↗</span></button>
        {!cameraEnabled && <button className="primary" type="button" onClick={startCamera}>Enter with camera</button>}
      </nav>

      <div className="coordinate coordinate-left">X 34.6937° / Y 135.5023°</div>
      <div className="coordinate coordinate-right">CURSED ENERGY // LIVE</div>
      {activeDomain === 'malevolent-shrine' && (
        <div className="model-credit">
          Shrine model by <a href="https://sketchfab.com/TSK_VITANK" target="_blank" rel="noreferrer">TSK_VITANK</a>
          {' · '}<a href="https://creativecommons.org/licenses/by-nc/4.0/" target="_blank" rel="noreferrer">CC BY-NC 4.0</a>
        </div>
      )}
    </main>
  )
}

export default App
