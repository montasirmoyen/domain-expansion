import { useCallback, useEffect, useRef, useState, type RefObject } from 'react'
import { FilesetResolver, HandLandmarker, DrawingUtils } from '@mediapipe/tasks-vision'
import type { DomainId, Landmark } from '../domain/domainTypes'
import { detectDomain, PredictionSmoother } from '../domain/gestureRecognition'

type Status = 'idle' | 'loading' | 'ready' | 'denied' | 'error'
type Args = { videoRef: RefObject<HTMLVideoElement | null>; overlayRef: RefObject<HTMLCanvasElement | null>; enabled: boolean }

export function useHandTracking({ videoRef, overlayRef, enabled }: Args) {
  const [status,setStatus] = useState<Status>('idle')
  const [prediction,setPrediction] = useState<DomainId | null>(null)
  const [handCount,setHandCount] = useState(0)
  const landmarker = useRef<HandLandmarker | null>(null)
  const stream = useRef<MediaStream | null>(null)
  const raf = useRef(0)
  const lastFrame = useRef(-1)
  const lastInference = useRef(0)
  const unmatchedFrames = useRef(0)
  const smoother = useRef(new PredictionSmoother())

  const enableCamera = useCallback(async () => {
    if (status === 'loading' || status === 'ready') return
    setStatus('loading')
    try {
      const media = await navigator.mediaDevices.getUserMedia({ video: { facingMode:'user', width:{ideal:960}, height:{ideal:600} }, audio:false })
      stream.current = media
      if (videoRef.current) { videoRef.current.srcObject=media; await videoRef.current.play() }
      const vision = await FilesetResolver.forVisionTasks(`${import.meta.env.BASE_URL}mediapipe`)
      landmarker.current = await HandLandmarker.createFromOptions(vision,{ baseOptions:{modelAssetPath:`${import.meta.env.BASE_URL}models/hand_landmarker.task`,delegate:'GPU'}, runningMode:'VIDEO', numHands:2, minHandDetectionConfidence:.55, minHandPresenceConfidence:.55, minTrackingConfidence:.5 })
      setStatus('ready')
    } catch (error) {
      const denied = error instanceof DOMException && (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError')
      setStatus(denied ? 'denied' : 'error')
      console.error(error)
    }
  },[status,videoRef])

  useEffect(() => {
    // `enabled` is controlled by a user gesture in the parent.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (enabled && status === 'idle') void enableCamera()
  },[enabled,status,enableCamera])

  useEffect(() => {
    if (status !== 'ready') return
    const loop = (now:number) => {
      const video=videoRef.current, canvas=overlayRef.current, tracker=landmarker.current
      if (video && canvas && tracker && video.readyState >= 2 && video.currentTime !== lastFrame.current && now-lastInference.current > 58) {
        lastFrame.current=video.currentTime; lastInference.current=now
        canvas.width=video.videoWidth; canvas.height=video.videoHeight
        const result=tracker.detectForVideo(video,now)
        const hands=result.landmarks as Landmark[][]
        setHandCount(hands.length)
        const detected=detectDomain(hands)
        const stable=smoother.current.push(detected)
        if (detected) unmatchedFrames.current=0
        else unmatchedFrames.current+=1
        if (stable) setPrediction(stable)
        else if (unmatchedFrames.current >= 6) {
          setPrediction(null)
          smoother.current.reset()
        }
        const ctx=canvas.getContext('2d')
        if (ctx) {
          ctx.clearRect(0,0,canvas.width,canvas.height)
          const drawing=new DrawingUtils(ctx)
          result.landmarks.forEach((hand) => {
            drawing.drawConnectors(hand,HandLandmarker.HAND_CONNECTIONS,{color:'#c8feff',lineWidth:2})
            drawing.drawLandmarks(hand,{color:'#ffffff',fillColor:'#6ee7e7',radius:2,lineWidth:1})
          })
        }
      }
      raf.current=requestAnimationFrame(loop)
    }
    raf.current=requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf.current)
  },[status,videoRef,overlayRef])

  useEffect(() => () => {
    cancelAnimationFrame(raf.current)
    stream.current?.getTracks().forEach((track)=>track.stop())
    landmarker.current?.close()
  },[])

  return {status,prediction,handCount,enableCamera}
}
