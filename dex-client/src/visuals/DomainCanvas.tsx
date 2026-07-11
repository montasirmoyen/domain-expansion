import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { useGLTF } from '@react-three/drei'
/* eslint-disable react-hooks/immutability */
import { Bloom, ChromaticAberration, EffectComposer, Noise, Vignette } from '@react-three/postprocessing'
import { AdditiveBlending, Box3, BufferAttribute, CanvasTexture, Color, DoubleSide, Fog, Group, Light, MathUtils, Mesh, Points, ShaderMaterial, Vector2, Vector3 } from 'three'
import { useMemo, useRef } from 'react'
import type { DomainId } from '../domain/domainTypes'

type SceneProps = { active: boolean }

const noise = (seed:number) => {
  const x=Math.sin(seed*91.3458+12.193)*43758.5453
  return x-Math.floor(x)
}

function FadeWorld({ active, children, speed=.7 }: SceneProps & { children: React.ReactNode; speed?:number }) {
  const ref=useRef<Group>(null)
  const strength=useRef(active?1:0)
  useFrame((_,delta)=>{
    strength.current=MathUtils.damp(strength.current,active?1:0,speed,delta)
    if (!ref.current) return
    ref.current.visible=strength.current>.004
    ref.current.traverse((object)=>{
      if (object instanceof Light) {
        if (object.userData.baseIntensity === undefined) object.userData.baseIntensity=object.intensity
        object.intensity=object.userData.baseIntensity*strength.current
      }
      const material=(object as Mesh).material
      if (!material) return
      const materials=Array.isArray(material)?material:[material]
      materials.forEach((mat)=>{
        if (mat.userData.baseOpacity === undefined) mat.userData.baseOpacity=mat.opacity
        mat.transparent=true
        mat.opacity=mat.userData.baseOpacity*strength.current
        mat.depthWrite=strength.current>.75
      })
    })
  })
  return <group ref={ref}>{children}</group>
}

function StarField({ count=1200, color='#a8eaff', radius=70, speed=.015 }) {
  const ref=useRef<Points>(null)
  const positions=useMemo(()=>{
    const a=new Float32Array(count*3)
    for(let i=0;i<count;i++){
      const r=radius*(.18+noise(i+count)*.82),t=noise(i*2.31+4)*Math.PI*2,p=Math.acos(2*noise(i*7.13+8)-1)
      a[i*3]=r*Math.sin(p)*Math.cos(t);a[i*3+1]=r*Math.cos(p);a[i*3+2]=r*Math.sin(p)*Math.sin(t)
    }
    return a
  },[count,radius])
  useFrame((_,delta)=>{if(ref.current){ref.current.rotation.y+=delta*speed;ref.current.rotation.z+=delta*speed*.16}})
  return <points ref={ref}><bufferGeometry><bufferAttribute attach="attributes-position" args={[positions,3]} /></bufferGeometry><pointsMaterial color={color} size={.16} sizeAttenuation transparent opacity={.82} depthWrite={false}/></points>
}

function NeutralWorld({active}:SceneProps){
  const rings=useRef<Group>(null)
  useFrame((s,d)=>{if(rings.current){rings.current.rotation.x=s.clock.elapsedTime*.05;rings.current.rotation.y+=d*.07}})
  return <FadeWorld active={active}><StarField count={700} color="#6de8e3"/><group ref={rings} position={[0,-1,-8]}>{[8,12,17,23].map((r,i)=><mesh key={r} rotation={[Math.PI/2+i*.3,i*.2,0]}><torusGeometry args={[r,.015,5,160]}/><meshBasicMaterial color={i%2?'#173c49':'#2f6970'} opacity={.42}/></mesh>)}</group></FadeWorld>
}

function BlackHoleMatter(){
  const material=useRef<ShaderMaterial>(null)
  const count=9200
  const data=useMemo(()=>{
    const positions=new Float32Array(count*3),radius=new Float32Array(count),angle=new Float32Array(count),speed=new Float32Array(count),tilt=new Float32Array(count),seed=new Float32Array(count)
    for(let i=0;i<count;i++){
      radius[i]=7+Math.pow(noise(i*2.17),.72)*35
      angle[i]=noise(i*5.41)*Math.PI*2
      speed[i]=.025+noise(i*8.93)*.055
      tilt[i]=(noise(i*3.73)-.5)*9
      seed[i]=noise(i*12.27)
    }
    return {positions,radius,angle,speed,tilt,seed}
  },[])
  useFrame((state)=>{if(material.current)material.current.uniforms.uTime.value=state.clock.elapsedTime})
  return <points>
    <bufferGeometry>
      <bufferAttribute attach="attributes-position" args={[data.positions,3]}/>
      <bufferAttribute attach="attributes-aRadius" args={[data.radius,1]}/>
      <bufferAttribute attach="attributes-aAngle" args={[data.angle,1]}/>
      <bufferAttribute attach="attributes-aSpeed" args={[data.speed,1]}/>
      <bufferAttribute attach="attributes-aTilt" args={[data.tilt,1]}/>
      <bufferAttribute attach="attributes-aSeed" args={[data.seed,1]}/>
    </bufferGeometry>
    <shaderMaterial ref={material} transparent depthWrite={false} blending={AdditiveBlending} uniforms={{uTime:{value:0}}}
      vertexShader={`
        uniform float uTime;
        attribute float aRadius; attribute float aAngle; attribute float aSpeed; attribute float aTilt; attribute float aSeed;
        varying float vHeat; varying float vSeed;
        void main(){
          float life=fract(aSeed+uTime*aSpeed);
          float plunge=pow(life,0.58);
          float radius=mix(aRadius,.18,plunge);
          float chaos=sin(aSeed*91.0+uTime*2.7+life*18.0);
          float theta=aAngle+life*15.0+pow(life,4.0)*34.0+uTime*.3+chaos*.2;
          vec3 p=vec3(cos(theta)*radius,aTilt*(1.0-life)*.42+chaos*(1.0-life)*1.1,sin(theta)*radius);
          vec4 mv=modelViewMatrix*vec4(p,1.0);
          gl_Position=projectionMatrix*mv;
          float surge=.72+.45*sin(uTime*5.2+aSeed*30.0);
          gl_PointSize=(2.0+7.0*pow(life,5.0))*surge*(85.0/-mv.z);
          vHeat=life; vSeed=aSeed;
        }`}
      fragmentShader={`
        varying float vHeat; varying float vSeed;
        void main(){
          float d=length(gl_PointCoord-.5); if(d>.5) discard;
          vec3 cold=mix(vec3(.14,.56,1.),vec3(.62,.2,1.),vSeed);
          vec3 hot=mix(cold,vec3(1.,.88,.64),pow(vHeat,4.));
          float alpha=smoothstep(.5,.04,d)*(1.-smoothstep(.92,1.,vHeat));
          gl_FragColor=vec4(hot,alpha);
        }`}/>
  </points>
}

function VoidInformationStorm(){
  const material=useRef<ShaderMaterial>(null)
  const count=4200
  const data=useMemo(()=>{
    const positions=new Float32Array(count*3),theta=new Float32Array(count),phi=new Float32Array(count),seed=new Float32Array(count),speed=new Float32Array(count)
    for(let i=0;i<count;i++){
      theta[i]=noise(i*3.17)*Math.PI*2
      phi[i]=Math.acos(2*noise(i*6.29)-1)
      seed[i]=noise(i*9.71)
      speed[i]=.045+noise(i*13.4)*.13
    }
    return {positions,theta,phi,seed,speed}
  },[])
  useFrame((state)=>{if(material.current)material.current.uniforms.uTime.value=state.clock.elapsedTime})
  return <points>
    <bufferGeometry>
      <bufferAttribute attach="attributes-position" args={[data.positions,3]}/>
      <bufferAttribute attach="attributes-aTheta" args={[data.theta,1]}/>
      <bufferAttribute attach="attributes-aPhi" args={[data.phi,1]}/>
      <bufferAttribute attach="attributes-aSeed" args={[data.seed,1]}/>
      <bufferAttribute attach="attributes-aSpeed" args={[data.speed,1]}/>
    </bufferGeometry>
    <shaderMaterial ref={material} transparent depthWrite={false} blending={AdditiveBlending} uniforms={{uTime:{value:0}}}
      vertexShader={`
        uniform float uTime;
        attribute float aTheta; attribute float aPhi; attribute float aSeed; attribute float aSpeed;
        varying float vLife; varying float vSeed;
        void main(){
          float life=fract(aSeed+uTime*aSpeed);
          float r=mix(72.0,.25,pow(life,.72));
          float twist=pow(life,3.0)*22.0+sin(uTime*3.0+aSeed*40.0)*1.5;
          float theta=aTheta+twist;
          float phi=aPhi+sin(life*18.0+aSeed*20.0)*.16;
          vec3 p=vec3(sin(phi)*cos(theta),cos(phi),sin(phi)*sin(theta))*r;
          p+=vec3(sin(aSeed*37.0+uTime*4.0),cos(aSeed*53.0+uTime*3.2),sin(aSeed*71.0+uTime*2.4))*mix(3.5,0.0,life);
          vec4 mv=modelViewMatrix*vec4(p,1.0);
          gl_Position=projectionMatrix*mv;
          float flash=step(.9,fract(aSeed*17.0+uTime*.8));
          gl_PointSize=(1.2+flash*5.0+pow(life,8.0)*8.0)*(75.0/-mv.z);
          vLife=life;vSeed=aSeed;
        }`}
      fragmentShader={`
        varying float vLife; varying float vSeed;
        void main(){
          float d=length(gl_PointCoord-.5);if(d>.5)discard;
          vec3 cyan=vec3(.2,.9,1.);vec3 violet=vec3(.58,.18,1.);vec3 white=vec3(1.);
          vec3 color=mix(cyan,violet,step(.48,vSeed));
          color=mix(color,white,pow(vLife,7.0));
          float alpha=smoothstep(.5,.02,d)*(1.-smoothstep(.96,1.,vLife));
          gl_FragColor=vec4(color,alpha);
        }`}/>
  </points>
}

function UnlimitedVoid({active}:SceneProps){
  const core=useRef<Group>(null)
  useFrame((state,delta)=>{
    if(!core.current)return
    core.current.rotation.y+=delta*.055
    core.current.rotation.z=Math.sin(state.clock.elapsedTime*.18)*.06
    const pulse=1+Math.sin(state.clock.elapsedTime*2.4)*.018
    core.current.scale.setScalar(pulse)
  })
  return <FadeWorld active={active} speed={8}>
    <StarField count={2300} radius={110} speed={-.025}/>
    <group ref={core} position={[0,-2,-19]}>
      <VoidInformationStorm/>
      <group rotation={[.78,0,-.08]}><BlackHoleMatter/></group>
    </group>
  </FadeWorld>
}

function CleaveParticles(){
  const ref=useRef<Points>(null),positionRef=useRef<BufferAttribute>(null)
  const count=3200
  const data=useMemo(()=>{
    const positions=new Float32Array(count*3),velocity=new Float32Array(count*3)
    for(let i=0;i<count;i++){
      positions[i*3]=(noise(i*2.2)-.5)*70;positions[i*3+1]=(noise(i*4.7)-.5)*45;positions[i*3+2]=-8-noise(i*7.4)*60
      velocity[i*3]=(noise(i*9.1)-.5)*.08;velocity[i*3+1]=.025+noise(i*11.2)*.12;velocity[i*3+2]=noise(i*14.6)*.05
    }
    return {positions,velocity}
  },[])
  useFrame((_,delta)=>{
    if(ref.current)ref.current.rotation.z+=delta*.018
    const a=data.positions
    for(let i=0;i<count;i++){
      a[i*3]+=data.velocity[i*3];a[i*3+1]+=data.velocity[i*3+1];a[i*3+2]+=data.velocity[i*3+2]
      if(a[i*3+1]>24){a[i*3+1]=-22;a[i*3]=(noise(i*18.1)-.5)*70;a[i*3+2]=-8-noise(i*21.8)*60}
    }
    if(positionRef.current)positionRef.current.needsUpdate=true
  })
  return <points ref={ref}><bufferGeometry><bufferAttribute ref={positionRef} attach="attributes-position" args={[data.positions,3]}/></bufferGeometry><pointsMaterial color="#ff3a24" size={.2} transparent opacity={.88} blending={AdditiveBlending} depthWrite={false}/></points>
}

function ShrineModel(){
  const {scene}=useGLTF('/shrine.glb')
  const ref=useRef<Group>(null)
  const model=useMemo(()=>{
    const clone=scene.clone(true)
    const bounds=new Box3().setFromObject(clone)
    const size=bounds.getSize(new Vector3())
    const center=bounds.getCenter(new Vector3())
    const scale=18/Math.max(size.x,size.y,size.z)
    clone.position.set(-center.x*scale,-bounds.min.y*scale,-center.z*scale)
    clone.scale.setScalar(scale)
    clone.traverse((object)=>{
      if(object instanceof Mesh){
        object.castShadow=true
        object.receiveShadow=true
        object.material=Array.isArray(object.material)
          ? object.material.map((material)=>material.clone())
          : object.material.clone()
      }
    })
    return clone
  },[scene])
  useFrame((state)=>{
    if(!ref.current)return
    ref.current.rotation.y=Math.sin(state.clock.elapsedTime*.18)*.07
    ref.current.position.y=-7+Math.sin(state.clock.elapsedTime*.45)*.08
  })
  return <group ref={ref} position={[0,-7,-24]} rotation={[0,.12,0]}><primitive object={model}/></group>
}

useGLTF.preload('/shrine.glb')

function MalevolentShrine({active}:SceneProps){
  const slashes=useRef<Group>(null),radialSlashes=useRef<Group>(null),cutGrid=useRef<Group>(null),shockwaves=useRef<Group>(null),debris=useRef<Group>(null)
  useFrame((state,delta)=>{
    const t=state.clock.elapsedTime
    if(slashes.current)slashes.current.children.forEach((slash,i)=>{
      const strike=Math.max(0,Math.sin(t*(2.8+(i%6)*.31)+i*1.73))
      slash.scale.x=.15+strike*1.85
      slash.scale.y=.08+Math.pow(strike,10)*3.4
      slash.position.x=(noise(i*3.8)-.5)*25+Math.sin(t*1.15+i)*3.5
      slash.position.y=(noise(i*5.7)-.5)*30+Math.cos(t*.9+i*.4)*1.8
    })
    if(radialSlashes.current){radialSlashes.current.rotation.z-=delta*.24;radialSlashes.current.children.forEach((slash,i)=>{const burst=Math.pow(Math.max(0,Math.sin(t*2.15+i*.57)),9);slash.scale.x=.1+burst*2.7;slash.scale.y=.12+burst*1.4})}
    if(cutGrid.current){cutGrid.current.rotation.z=Math.sin(t*.45)*.08;cutGrid.current.children.forEach((line,i)=>{const snap=Math.pow(Math.max(0,Math.sin(t*(3.4+(i%3)*.35)+i)),12);line.scale.x=.25+snap*2.2;line.scale.y=.1+snap*2.8})}
    if(shockwaves.current)shockwaves.current.children.forEach((ring,i)=>{const cycle=(t*.78+i*.21)%1;ring.scale.setScalar(.25+cycle*4.4);ring.rotation.z+=delta*(i%2?-.28:.28)})
    if(debris.current){debris.current.rotation.y+=delta*.17;debris.current.children.forEach((piece,i)=>{piece.rotation.x+=delta*(.35+noise(i)*1.2);piece.rotation.z-=delta*(.25+noise(i+4)*.9);piece.position.y+=Math.sin(t*2+i)*.004})}
  })
  return <FadeWorld active={active} speed={1.55}>
    <ambientLight intensity={.7} color="#ff180d"/><pointLight position={[0,0,-8]} intensity={190} color="#ff2414"/><pointLight position={[-14,8,-18]} intensity={85} color="#ff8060"/>
    <ShrineModel/>
    <CleaveParticles/>
    <group ref={slashes}>{Array.from({length:64},(_,i)=><group key={i} position={[0,(noise(i*4.2)-.5)*28,-5-noise(i*8.1)*52]} rotation={[(noise(i*2.3)-.5)*.7,(noise(i*5.2)-.5)*.8,(noise(i*9.7)-.5)*Math.PI]}>
      <mesh><planeGeometry args={[23,.035]}/><meshBasicMaterial color="#fff4e8" side={DoubleSide} transparent opacity={.98} blending={AdditiveBlending}/></mesh>
      <mesh><planeGeometry args={[25,.42]}/><meshBasicMaterial color={i%4===0?'#ffb4a6':'#ff2115'} side={DoubleSide} transparent opacity={.28} blending={AdditiveBlending} depthWrite={false}/></mesh>
    </group>)}</group>
    <group ref={radialSlashes} position={[0,-2,-17]}>{Array.from({length:28},(_,i)=><group key={i} rotation={[0,0,i/28*Math.PI*2]}><mesh position={[11,0,-i%4]}><planeGeometry args={[22,.06]}/><meshBasicMaterial color={i%3===0?'#fff9f5':'#ff2518'} transparent opacity={.72} blending={AdditiveBlending} depthWrite={false}/></mesh></group>)}</group>
    <group ref={cutGrid} position={[0,0,-32]}>{Array.from({length:18},(_,i)=><mesh key={i} position={[(i%6-2.5)*7,(Math.floor(i/6)-1)*10,0]} rotation={[0,0,i%2?Math.PI/2:0]}><planeGeometry args={[42,.055]}/><meshBasicMaterial color={i%5===0?'#ffffff':'#b80c08'} transparent opacity={.4} blending={AdditiveBlending}/></mesh>)}</group>
    <group ref={shockwaves} position={[0,-6.85,-24]} rotation={[Math.PI/2,0,0]}>{[0,1,2,3,4].map(i=><mesh key={i}><torusGeometry args={[5,.055,7,160]}/><meshBasicMaterial color={i===0?'#ffffff':'#ff2517'} transparent opacity={.55} blending={AdditiveBlending}/></mesh>)}</group>
    <group ref={debris} position={[0,-5,-20]}>{Array.from({length:90},(_,i)=><mesh key={i} position={[(noise(i*3.4)-.5)*42,noise(i*5.2)*20,(noise(i*8.8)-.5)*38]} scale={.1+noise(i*12.3)*.85}><tetrahedronGeometry args={[1,0]}/><meshStandardMaterial color={i%5===0?'#ff3a20':'#210303'} emissive={i%5===0?'#c51a0d':'#000000'} emissiveIntensity={2.5}/></mesh>)}</group>
    <mesh position={[0,-7,-23]} rotation={[-Math.PI/2,0,0]}><ringGeometry args={[3,34,96]}/><meshBasicMaterial color="#260201" transparent opacity={.72} side={DoubleSide}/></mesh>
  </FadeWorld>
}

function HandSculpture({position,rotation,scale=1}:{position:[number,number,number];rotation:[number,number,number];scale?:number}){
  return <group position={position} rotation={rotation} scale={scale}><mesh><boxGeometry args={[3.3,4,.8]}/><meshStandardMaterial color="#5b286c" roughness={.8}/></mesh>{[-1.25,-.42,.42,1.25].map((x,i)=><mesh key={x} position={[x,3+i*.18,0]}><capsuleGeometry args={[.4,4.2-i*.25,5,9]}/><meshStandardMaterial color="#75378a" roughness={.75}/></mesh>)}<mesh position={[-2.1,.4,0]} rotation={[0,0,-.65]}><capsuleGeometry args={[.43,3.1,5,9]}/><meshStandardMaterial color="#75378a"/></mesh></group>
}
function SelfEmbodiment({active}:SceneProps){
  const flower=useRef<Group>(null)
  useFrame((s,d)=>{if(flower.current){flower.current.rotation.y+=d*.035;flower.current.children.forEach((c,i)=>{c.rotation.z+=Math.sin(s.clock.elapsedTime*.6+i)*.0008})}})
  return <FadeWorld active={active} speed={1.2}><ambientLight color="#8f3ab5" intensity={1.4}/><pointLight position={[0,0,-5]} color="#d457ff" intensity={80}/><group ref={flower} position={[0,-2,-14]}>{Array.from({length:14},(_,i)=>{const a=i/14*Math.PI*2,r=10+(i%3)*3;return <HandSculpture key={i} position={[Math.cos(a)*r,Math.sin(a)*r,Math.sin(a*2)*3]} rotation={[0,0,a-Math.PI/2]} scale={.68+(i%3)*.13}/>})}<HandSculpture position={[0,-1,2]} rotation={[0,0,0]} scale={1.55}/></group><StarField count={800} color="#b845e8" radius={45}/></FadeWorld>
}

function makeLabel(text:string,fg='#ffe368',bg='#1b0e00'){
  const canvas=document.createElement('canvas');canvas.width=1024;canvas.height=256;const c=canvas.getContext('2d')!;c.fillStyle=bg;c.fillRect(0,0,1024,256);c.strokeStyle=fg;c.lineWidth=12;c.strokeRect(10,10,1004,236);c.fillStyle=fg;c.font='bold 132px sans-serif';c.textAlign='center';c.textBaseline='middle';c.fillText(text,512,135);return canvas
}
function IdleDeathGamble({active}:SceneProps){
  const wheels=useRef<Group>(null)
  const texture=useMemo(()=>new CanvasTexture(makeLabel('7  7  7')),[])
  useFrame((s,d)=>{if(wheels.current){wheels.current.rotation.y+=d*.12;wheels.current.position.y=Math.sin(s.clock.elapsedTime*1.7)*.35}})
  return <FadeWorld active={active} speed={1.4}><ambientLight intensity={.7} color="#ffb820"/><pointLight position={[0,8,2]} intensity={110} color="#ffd633"/><mesh position={[0,2,-18]}><planeGeometry args={[25,6]}/><meshBasicMaterial map={texture} toneMapped={false}/></mesh><group ref={wheels} position={[0,-2,-8]}>{[-7,0,7].map((x,i)=><mesh key={x} position={[x,0,-i]} rotation={[Math.PI/2,0,0]}><torusGeometry args={[3.2,.55,10,60]}/><meshStandardMaterial color={i===1?'#ff335f':'#f7bd20'} emissive={i===1?'#ff174d':'#b36d00'} emissiveIntensity={2}/></mesh>)}</group>{Array.from({length:16},(_,i)=><mesh key={i} position={[(i%4-1.5)*8,-6,Math.floor(i/4)*-12+4]}><boxGeometry args={[5,.25,9]}/><meshStandardMaterial color={i%3===0?'#b8911f':'#4c3310'} emissive={i%3===0?'#886000':'#000'} emissiveIntensity={1}/></mesh>)}<StarField count={1700} color="#ffd31c" radius={65} speed={.05}/></FadeWorld>
}

function Katana({x,z,rotation}:{x:number;z:number;rotation:number}){return <group position={[x,-4,z]} rotation={[0,rotation,.08]}><mesh position={[0,2.6,0]}><boxGeometry args={[.15,5.2,.12]}/><meshStandardMaterial color="#e7e3df" metalness={.8} roughness={.22}/></mesh><mesh><boxGeometry args={[1.1,.12,.25]}/><meshStandardMaterial color="#ffb4d8" emissive="#722348" emissiveIntensity={1.5}/></mesh><mesh position={[0,-.8,0]}><cylinderGeometry args={[.15,.18,1.6,8]}/><meshStandardMaterial color="#22121b"/></mesh></group>}
function AuthenticLove({active}:SceneProps){
  const field=useRef<Group>(null)
  const blades=useMemo(()=>Array.from({length:75},(_,i)=>({x:(noise(i*3.1)-.5)*58,z:-noise(i*7.7)*70+9,rotation:noise(i*11.3)*6.28})),[])
  useFrame((s)=>{if(field.current){field.current.rotation.y=Math.sin(s.clock.elapsedTime*.11)*.025}})
  return <FadeWorld active={active} speed={1.25}><ambientLight intensity={.5} color="#ffc0e4"/><directionalLight position={[-8,12,4]} intensity={4} color="#ffb3d8"/><group ref={field}>{blades.map((b,i)=><Katana key={i} {...b}/>)}</group>{[-16,16].map((x)=><group key={x} position={[x,4,-23]} rotation={[0,0,x>0?.25:-.25]}><mesh><boxGeometry args={[2,22,2]}/><meshStandardMaterial color="#351629" roughness={.9}/></mesh><mesh><boxGeometry args={[13,2,2]}/><meshStandardMaterial color="#351629" roughness={.9}/></mesh></group>)}<mesh position={[0,1,-25]}><torusGeometry args={[9,.11,8,120]}/><meshBasicMaterial color="#ff9ccf" opacity={.75}/></mesh><StarField count={900} color="#ff8fc9" radius={50}/></FadeWorld>
}

function MemoryMotes(){
  const ref=useRef<Points>(null),positionRef=useRef<BufferAttribute>(null)
  const count=1500
  const positions=useMemo(()=>{const a=new Float32Array(count*3);for(let i=0;i<count;i++){a[i*3]=(noise(i*2.8)-.5)*62;a[i*3+1]=-8+noise(i*5.1)*32;a[i*3+2]=-5-noise(i*8.4)*78}return a},[])
  useFrame((state)=>{
    for(let i=0;i<count;i++){
      positions[i*3]+=.006*Math.sin(state.clock.elapsedTime*.7+i)
      positions[i*3+1]+=.012+noise(i*4.2)*.025
      if(positions[i*3+1]>25)positions[i*3+1]=-8
    }
    if(positionRef.current)positionRef.current.needsUpdate=true
    if(ref.current)ref.current.rotation.y=Math.sin(state.clock.elapsedTime*.08)*.035
  })
  return <points ref={ref}><bufferGeometry><bufferAttribute ref={positionRef} attach="attributes-position" args={[positions,3]}/></bufferGeometry><pointsMaterial color="#ffc18d" size={.22} transparent opacity={.72} blending={AdditiveBlending} depthWrite={false}/></points>
}

function YujiDomain({active}:SceneProps){
  return <FadeWorld active={active} speed={.75}><hemisphereLight args={['#ffc49b','#2a2631',2.4]}/><directionalLight position={[-8,18,8]} intensity={3.4} color="#ffb17d"/><MemoryMotes/><mesh position={[9,-4.25,-22]}><boxGeometry args={[13,.5,62]}/><meshStandardMaterial color="#7e6a5e" roughness={.9}/></mesh>{Array.from({length:13},(_,i)=><group key={i} position={[13,-1,-4-i*6]}><mesh><cylinderGeometry args={[.09,.12,6,8]}/><meshStandardMaterial color="#26252b"/></mesh><mesh position={[0,3,0]}><sphereGeometry args={[.32,12,12]}/><meshBasicMaterial color="#ffd59a"/></mesh></group>)}{Array.from({length:22},(_,i)=><mesh key={i} position={[(i%2?1:-1)*(15+(i%4)*4),-1,-8-Math.floor(i/2)*9]}><boxGeometry args={[5,7+(i%3)*3,6]}/><meshStandardMaterial color={i%2?'#382f31':'#51413a'} roughness={.92}/></mesh>)}<StarField count={700} color="#ff9d67" radius={75} speed={.007}/></FadeWorld>
}

const ENVIRONMENTS: Record<DomainId,{background:string;fog:string;near:number;far:number}> = {
  neutral:{background:'#030307',fog:'#030307',near:20,far:90},
  'unlimited-void':{background:'#000006',fog:'#02000e',near:28,far:115},
  'malevolent-shrine':{background:'#100101',fog:'#170201',near:18,far:72},
  'self-embodiment':{background:'#020003',fog:'#050007',near:18,far:60},
  'idle-death-gamble':{background:'#060300',fog:'#0d0700',near:25,far:85},
  'authentic-mutual-love':{background:'#09030a',fog:'#160914',near:18,far:78},
  'yuji-domain':{background:'#241611',fog:'#8f4c34',near:25,far:92},
}

function DynamicEnvironment({active}:{active:DomainId}){
  const {scene}=useThree()
  const background=useRef(new Color(ENVIRONMENTS.neutral.background))
  const fog=useRef(new Fog(ENVIRONMENTS.neutral.fog,20,90))
  useFrame((_,delta)=>{
    const target=ENVIRONMENTS[active]
    background.current.lerp(new Color(target.background),1-Math.exp(-delta*1.2))
    fog.current.color.lerp(new Color(target.fog),1-Math.exp(-delta*1.2))
    fog.current.near=MathUtils.damp(fog.current.near,target.near,1.2,delta)
    fog.current.far=MathUtils.damp(fog.current.far,target.far,1.2,delta)
    scene.background=background.current
    scene.fog=fog.current
  })
  return null
}

function CameraRig({active}:{active:DomainId}){
  const {camera}=useThree()
  useFrame((state,delta)=>{
    const t=state.clock.elapsedTime
    const targetZ=active==='malevolent-shrine'?18:active==='self-embodiment'?19:active==='yuji-domain'?17:20
    const cleaveImpact=active==='malevolent-shrine'?Math.pow(Math.max(0,Math.sin(t*2.15)),12):0
    const shrineShakeX=active==='malevolent-shrine'?Math.sin(t*31)*(.13+cleaveImpact*.32)+Math.sin(t*47)*.08:0
    const shrineShakeY=active==='malevolent-shrine'?Math.sin(t*37)*(.1+cleaveImpact*.24):0
    const voidDriftY=active==='unlimited-void'?Math.sin(t*.17)*1.05+Math.sin(t*.53)*.22:0
    const voidPush=active==='unlimited-void'?Math.sin(t*.31)*.55:0
    camera.position.z=MathUtils.damp(camera.position.z,targetZ+voidPush-cleaveImpact*.55,active==='malevolent-shrine'?5:1.5,delta)
    camera.position.x=MathUtils.damp(camera.position.x,Math.sin(t*.11)*(active==='unlimited-void'?2.2:.65)+shrineShakeX,active==='malevolent-shrine'?8:1.2,delta)
    camera.position.y=MathUtils.damp(camera.position.y,(active==='yuji-domain'?2:0)+shrineShakeY+voidDriftY,active==='malevolent-shrine'?8:1.2,delta)
    camera.lookAt(0,active==='yuji-domain'?-1:0,-12)
  })
  return null
}

export function DomainCanvas({active}: {active:DomainId; previous:DomainId; transitionKey:number}){
  return <Canvas className="world-canvas" dpr={[1,1.75]} gl={{antialias:true,powerPreference:'high-performance'}} camera={{fov:55,near:.1,far:220,position:[0,0,20]}}>
    <DynamicEnvironment active={active}/>
    <CameraRig active={active}/>
    <NeutralWorld active={active==='neutral'}/>
    <UnlimitedVoid active={active==='unlimited-void'}/>
    <MalevolentShrine active={active==='malevolent-shrine'}/>
    <SelfEmbodiment active={active==='self-embodiment'}/>
    <IdleDeathGamble active={active==='idle-death-gamble'}/>
    <AuthenticLove active={active==='authentic-mutual-love'}/>
    <YujiDomain active={active==='yuji-domain'}/>
    <EffectComposer multisampling={0}>
      <Bloom intensity={1.58} luminanceThreshold={.57} luminanceSmoothing={.3} mipmapBlur/>
      <ChromaticAberration offset={new Vector2(.00045,.00065)} radialModulation modulationOffset={.22}/>
      <Noise opacity={.035}/><Vignette eskil={false} offset={.18} darkness={.65}/>
    </EffectComposer>
  </Canvas>
}
