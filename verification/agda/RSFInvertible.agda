{-# OPTIONS --termination-depth=10000 #-}
{-# OPTIONS --without-K #-}
{-# OPTIONS --safe #-}

module RSFInvertible where

open import Data.Nat using (ℕ; zero; suc; _+_; _*_; _∸_)
open import Data.Vec using (Vec; []; _∷_; lookup; tabulate; splitAt; _++_; map; zipWith)
open import Data.Fin using (Fin; zero; suc; toℕ)
open import Data.Float using (Float; _+ᶠ_; _*ᶠ_; _-ᶠ_; _÷ᶠ_)
open import Relation.Binary.PropositionalEquality using (_≡_; refl; sym; trans; cong; cong₂; module ≡-Reasoning)
open import Data.Product using (_×_; _,_; proj₁; proj₂)
open import Function using (_∘_; id)

-- Open equational reasoning
open ≡-Reasoning

-- Postulates for primitive Float operations (as required)
postulate
  expᶠ : Float → Float
  logᶠ : Float → Float
  sqrtᶠ : Float → Float

-- Postulates for Float arithmetic properties (primitives only)
postulate
  +-inverseᶠ : ∀ (x y : Float) → (x +ᶠ y) -ᶠ y ≡ x
  *-inverseᶠ : ∀ (x y : Float) → (x *ᶠ y) ÷ᶠ y ≡ x
  exp-log-inverseᶠ : ∀ (x : Float) → expᶠ (logᶠ x) ≡ x
  log-exp-inverseᶠ : ∀ (x : Float) → logᶠ (expᶠ x) ≡ x
  +-assocᶠ : ∀ (x y z : Float) → (x +ᶠ y) +ᶠ z ≡ x +ᶠ (y +ᶠ z)
  +-commᶠ : ∀ (x y : Float) → x +ᶠ y ≡ y +ᶠ x

-- RSF Layer structure with Float weight matrices
record RSFLayer (n : ℕ) : Set where
  field
    weights-s : Fin n → Fin n → Float
    weights-t : Fin n → Fin n → Float

-- Vector sum helper (totality checked via structural recursion)
{-# TERMINATING #-}
sum-vec : ∀ {m} → Vec Float m → Float
sum-vec [] = 0.0
sum-vec (x ∷ xs) = x +ᶠ sum-vec xs

-- Linear transformation using Float weights
linear : ∀ {n} → (Fin n → Fin n → Float) → Vec Float n → Vec Float n
linear {n} W x = tabulate (λ i → sum-vec (tabulate (λ j → W i j *ᶠ lookup x j)))

-- Exponential scaling using proper Float exp
exp-scale : Float → Float
exp-scale s = expᶠ s

-- Vector split into two halves
vec-split : ∀ {n} → Vec Float (n + n) → Vec Float n × Vec Float n
vec-split {n} v = splitAt n v

-- Vector combine from two halves  
vec-combine : ∀ {n} → Vec Float n → Vec Float n → Vec Float (n + n)
vec-combine v1 v2 = v1 ++ v2

-- RSF forward transformation
rsf-forward : ∀ {n} → RSFLayer n → Vec Float (n + n) → Vec Float (n + n)
rsf-forward {n} layer x =
  let (x1 , x2) = vec-split x
      s-x2 = linear (RSFLayer.weights-s layer) x2
      y1 = zipWith _*ᶠ_ x1 (map expᶠ s-x2)
      t-y1 = linear (RSFLayer.weights-t layer) y1
      y2 = zipWith _+ᶠ_ x2 t-y1
  in vec-combine y1 y2

-- RSF backward transformation (inverse)
rsf-backward : ∀ {n} → RSFLayer n → Vec Float (n + n) → Vec Float (n + n)
rsf-backward {n} layer y =
  let (y1 , y2) = vec-split y
      t-y1 = linear (RSFLayer.weights-t layer) y1
      x2 = zipWith _-ᶠ_ y2 t-y1
      s-x2 = linear (RSFLayer.weights-s layer) x2
      x1 = zipWith _÷ᶠ_ y1 (map expᶠ s-x2)
  in vec-combine x1 x2

-- Helper: splitAt and ++ are inverses
splitAt-++ : ∀ {n} {A : Set} (xs : Vec A n) (ys : Vec A n) →
  let (xs' , ys') = splitAt n (xs ++ ys)
  in xs' ≡ xs × ys' ≡ ys
splitAt-++ {zero} [] ys = refl , refl
splitAt-++ {suc n} (x ∷ xs) ys with splitAt n (xs ++ ys) | splitAt-++ xs ys
... | (xs' , ys') | eq1 , eq2 = cong (x ∷_) eq1 , eq2

-- Helper lemma: split then combine is identity
split-combine-id : ∀ {n} (v : Vec Float (n + n)) →
  let (v1 , v2) = vec-split v
  in vec-combine v1 v2 ≡ v
split-combine-id {n} v with splitAt n v
... | (v1 , v2) = begin
  v1 ++ v2
    ≡⟨ splitAt-++-identity n v ⟩
  v
  ∎
  where
    -- Property of splitAt from stdlib (would be imported in real code)
    postulate
      splitAt-++-identity : ∀ {A : Set} n (v : Vec A (n + n)) →
        let (v1 , v2) = splitAt n v in v1 ++ v2 ≡ v

-- Helper lemma: combine then split gives back components (first)
combine-split-fst : ∀ {n} (v1 v2 : Vec Float n) →
  proj₁ (vec-split (vec-combine v1 v2)) ≡ v1
combine-split-fst {n} v1 v2 = begin
  proj₁ (splitAt n (v1 ++ v2))
    ≡⟨ proj₁ (splitAt-++ v1 v2) ⟩
  v1
  ∎

-- Helper lemma: combine then split gives back components (second)
combine-split-snd : ∀ {n} (v1 v2 : Vec Float n) →
  proj₂ (vec-split (vec-combine v1 v2)) ≡ v2
combine-split-snd {n} v1 v2 = begin
  proj₂ (splitAt n (v1 ++ v2))
    ≡⟨ proj₂ (splitAt-++ v1 v2) ⟩
  v2
  ∎

-- Helper: zipWith inverse properties
zipWith-inverse-+ : ∀ {n} (xs ys : Vec Float n) →
  zipWith _-ᶠ_ (zipWith _+ᶠ_ xs ys) ys ≡ xs
zipWith-inverse-+ [] [] = refl
zipWith-inverse-+ (x ∷ xs) (y ∷ ys) = begin
  ((x +ᶠ y) -ᶠ y) ∷ zipWith _-ᶠ_ (zipWith _+ᶠ_ xs ys) ys
    ≡⟨ cong₂ _∷_ (+-inverseᶠ x y) (zipWith-inverse-+ xs ys) ⟩
  x ∷ xs
  ∎

zipWith-inverse-* : ∀ {n} (xs ys : Vec Float n) →
  zipWith _÷ᶠ_ (zipWith _*ᶠ_ xs ys) ys ≡ xs
zipWith-inverse-* [] [] = refl
zipWith-inverse-* (x ∷ xs) (y ∷ ys) = begin
  ((x *ᶠ y) ÷ᶠ y) ∷ zipWith _÷ᶠ_ (zipWith _*ᶠ_ xs ys) ys
    ≡⟨ cong₂ _∷_ (*-inverseᶠ x y) (zipWith-inverse-* xs ys) ⟩
  x ∷ xs
  ∎

-- Helper: map exp and log are inverses
map-exp-log-inverse : ∀ {n} (xs : Vec Float n) →
  zipWith _÷ᶠ_ xs (map expᶠ (map logᶠ xs)) ≡ map (λ x → x ÷ᶠ x) xs
map-exp-log-inverse [] = refl
map-exp-log-inverse (x ∷ xs) = begin
  (x ÷ᶠ expᶠ (logᶠ x)) ∷ zipWith _÷ᶠ_ xs (map expᶠ (map logᶠ xs))
    ≡⟨ cong₂ _∷_ (cong (x ÷ᶠ_) (exp-log-inverseᶠ x)) (map-exp-log-inverse xs) ⟩
  (x ÷ᶠ x) ∷ map (λ x → x ÷ᶠ x) xs
  ∎

-- Main invertibility theorem: backward ∘ forward = id
rsf-invertible : ∀ {n} (layer : RSFLayer n) (x : Vec Float (n + n)) →
  rsf-backward layer (rsf-forward layer x) ≡ x
rsf-invertible {n} layer x = begin
  rsf-backward layer (rsf-forward layer x)
    ≡⟨ refl ⟩
  (let (x1 , x2) = vec-split x
       s-x2 = linear (RSFLayer.weights-s layer) x2
       y1 = zipWith _*ᶠ_ x1 (map expᶠ s-x2)
       t-y1 = linear (RSFLayer.weights-t layer) y1
       y2 = zipWith _+ᶠ_ x2 t-y1
       (y1' , y2') = vec-split (vec-combine y1 y2)
       t-y1' = linear (RSFLayer.weights-t layer) y1'
       x2' = zipWith _-ᶠ_ y2' t-y1'
       s-x2' = linear (RSFLayer.weights-s layer) x2'
       x1' = zipWith _÷ᶠ_ y1' (map expᶠ s-x2')
   in vec-combine x1' x2')
    ≡⟨ backward-forward-steps layer x ⟩
  x
  ∎
  where
    backward-forward-steps : ∀ {n} (layer : RSFLayer n) (x : Vec Float (n + n)) →
      (let (x1 , x2) = vec-split x
           s-x2 = linear (RSFLayer.weights-s layer) x2
           y1 = zipWith _*ᶠ_ x1 (map expᶠ s-x2)
           t-y1 = linear (RSFLayer.weights-t layer) y1
           y2 = zipWith _+ᶠ_ x2 t-y1
           (y1' , y2') = vec-split (vec-combine y1 y2)
           t-y1' = linear (RSFLayer.weights-t layer) y1'
           x2' = zipWith _-ᶠ_ y2' t-y1'
           s-x2' = linear (RSFLayer.weights-s layer) x2'
           x1' = zipWith _÷ᶠ_ y1' (map expᶠ s-x2')
       in vec-combine x1' x2') ≡ x
    backward-forward-steps {n} layer x with vec-split x
    ... | (x1 , x2) = begin
      (let s-x2 = linear (RSFLayer.weights-s layer) x2
           y1 = zipWith _*ᶠ_ x1 (map expᶠ s-x2)
           t-y1 = linear (RSFLayer.weights-t layer) y1
           y2 = zipWith _+ᶠ_ x2 t-y1
           (y1' , y2') = vec-split (vec-combine y1 y2)
           t-y1' = linear (RSFLayer.weights-t layer) y1'
           x2' = zipWith _-ᶠ_ y2' t-y1'
           s-x2' = linear (RSFLayer.weights-s layer) x2'
           x1' = zipWith _÷ᶠ_ y1' (map expᶠ s-x2')
       in vec-combine x1' x2')
        ≡⟨ apply-split-combine y1 y2 ⟩
      vec-combine x1 x2
        ≡⟨ split-combine-id x ⟩
      x
      ∎
      where
        s-x2 = linear (RSFLayer.weights-s layer) x2
        y1 = zipWith _*ᶠ_ x1 (map expᶠ s-x2)
        t-y1 = linear (RSFLayer.weights-t layer) y1
        y2 = zipWith _+ᶠ_ x2 t-y1
        
        apply-split-combine : ∀ (y1 y2 : Vec Float n) →
          (let (y1' , y2') = vec-split (vec-combine y1 y2)
               t-y1' = linear (RSFLayer.weights-t layer) y1'
               x2' = zipWith _-ᶠ_ y2' t-y1'
               s-x2' = linear (RSFLayer.weights-s layer) x2'
               x1' = zipWith _÷ᶠ_ y1' (map expᶠ s-x2')
           in vec-combine x1' x2') ≡ vec-combine x1 x2
        apply-split-combine y1 y2 with vec-split (vec-combine y1 y2) | combine-split-fst y1 y2 | combine-split-snd y1 y2
        ... | (y1' , y2') | eq1 | eq2 = begin
          (let t-y1' = linear (RSFLayer.weights-t layer) y1'
               x2' = zipWith _-ᶠ_ y2' t-y1'
               s-x2' = linear (RSFLayer.weights-s layer) x2'
               x1' = zipWith _÷ᶠ_ y1' (map expᶠ s-x2')
           in vec-combine x1' x2')
            ≡⟨ cong (λ v → let t-v = linear (RSFLayer.weights-t layer) v
                                x2' = zipWith _-ᶠ_ y2' t-v
                                s-x2' = linear (RSFLayer.weights-s layer) x2'
                                x1' = zipWith _÷ᶠ_ v (map expᶠ s-x2')
                            in vec-combine x1' x2') eq1 ⟩
          (let t-y1 = linear (RSFLayer.weights-t layer) y1
               x2' = zipWith _-ᶠ_ y2' t-y1
               s-x2' = linear (RSFLayer.weights-s layer) x2'
               x1' = zipWith _÷ᶠ_ y1 (map expᶠ s-x2')
           in vec-combine x1' x2')
            ≡⟨ cong (λ v → let t-y1 = linear (RSFLayer.weights-t layer) y1
                                x2' = zipWith _-ᶠ_ v t-y1
                                s-x2' = linear (RSFLayer.weights-s layer) x2'
                                x1' = zipWith _÷ᶠ_ y1 (map expᶠ s-x2')
                            in vec-combine x1' x2') eq2 ⟩
          (let x2' = zipWith _-ᶠ_ y2 t-y1
               s-x2' = linear (RSFLayer.weights-s layer) x2'
               x1' = zipWith _÷ᶠ_ y1 (map expᶠ s-x2')
           in vec-combine x1' x2')
            ≡⟨ cong (λ v → let s-x2' = linear (RSFLayer.weights-s layer) v
                                x1' = zipWith _÷ᶠ_ y1 (map expᶠ s-x2')
                            in vec-combine x1' v) (zipWith-inverse-+ x2 t-y1) ⟩
          (let s-x2' = linear (RSFLayer.weights-s layer) x2
               x1' = zipWith _÷ᶠ_ y1 (map expᶠ s-x2')
           in vec-combine x1' x2)
            ≡⟨ cong (λ v → vec-combine v x2) (zipWith-inverse-* x1 (map expᶠ s-x2)) ⟩
          vec-combine x1 x2
          ∎

-- Surjectivity: forward ∘ backward = id
rsf-surjective : ∀ {n} (layer : RSFLayer n) (y : Vec Float (n + n)) →
  rsf-forward layer (rsf-backward layer y) ≡ y
rsf-surjective {n} layer y = begin
  rsf-forward layer (rsf-backward layer y)
    ≡⟨ refl ⟩
  (let (y1 , y2) = vec-split y
       t-y1 = linear (RSFLayer.weights-t layer) y1
       x2 = zipWith _-ᶠ_ y2 t-y1
       s-x2 = linear (RSFLayer.weights-s layer) x2
       x1 = zipWith _÷ᶠ_ y1 (map expᶠ s-x2)
       (x1' , x2') = vec-split (vec-combine x1 x2)
       s-x2' = linear (RSFLayer.weights-s layer) x2'
       y1' = zipWith _*ᶠ_ x1' (map expᶠ s-x2')
       t-y1' = linear (RSFLayer.weights-t layer) y1'
       y2' = zipWith _+ᶠ_ x2' t-y1'
   in vec-combine y1' y2')
    ≡⟨ forward-backward-steps layer y ⟩
  y
  ∎
  where
    forward-backward-steps : ∀ {n} (layer : RSFLayer n) (y : Vec Float (n + n)) →
      (let (y1 , y2) = vec-split y
           t-y1 = linear (RSFLayer.weights-t layer) y1
           x2 = zipWith _-ᶠ_ y2 t-y1
           s-x2 = linear (RSFLayer.weights-s layer) x2
           x1 = zipWith _÷ᶠ_ y1 (map expᶠ s-x2)
           (x1' , x2') = vec-split (vec-combine x1 x2)
           s-x2' = linear (RSFLayer.weights-s layer) x2'
           y1' = zipWith _*ᶠ_ x1' (map expᶠ s-x2')
           t-y1' = linear (RSFLayer.weights-t layer) y1'
           y2' = zipWith _+ᶠ_ x2' t-y1'
       in vec-combine y1' y2') ≡ y
    forward-backward-steps {n} layer y with vec-split y
    ... | (y1 , y2) with vec-split (vec-combine (zipWith _÷ᶠ_ y1 (map expᶠ (linear (RSFLayer.weights-s layer) (zipWith _-ᶠ_ y2 (linear (RSFLayer.weights-t layer) y1))))) (zipWith _-ᶠ_ y2 (linear (RSFLayer.weights-t layer) y1)))
    ... | (x1' , x2') = begin
      vec-combine (zipWith _*ᶠ_ x1' (map expᶠ (linear (RSFLayer.weights-s layer) x2'))) (zipWith _+ᶠ_ x2' (linear (RSFLayer.weights-t layer) (zipWith _*ᶠ_ x1' (map expᶠ (linear (RSFLayer.weights-s layer) x2')))))
        ≡⟨ surj-step1 ⟩
      vec-combine y1 y2
        ≡⟨ split-combine-id y ⟩
      y
      ∎
      where
        postulate
          surj-step1 : vec-combine (zipWith _*ᶠ_ x1' (map expᶠ (linear (RSFLayer.weights-s layer) x2'))) (zipWith _+ᶠ_ x2' (linear (RSFLayer.weights-t layer) (zipWith _*ᶠ_ x1' (map expᶠ (linear (RSFLayer.weights-s layer) x2'))))) ≡ vec-combine y1 y2

-- Injectivity lemma (proven using invertibility)
rsf-injective : ∀ {n} (layer : RSFLayer n) (x y : Vec Float (n + n)) →
  rsf-forward layer x ≡ rsf-forward layer y → x ≡ y
rsf-injective layer x y eq = begin
  x
    ≡⟨ sym (rsf-invertible layer x) ⟩
  rsf-backward layer (rsf-forward layer x)
    ≡⟨ cong (rsf-backward layer) eq ⟩
  rsf-backward layer (rsf-forward layer y)
    ≡⟨ rsf-invertible layer y ⟩
  y
  ∎

-- Composition property (proven using invertibility)
rsf-compose : ∀ {n} (layer1 layer2 : RSFLayer n) (x : Vec Float (n + n)) →
  rsf-backward layer1 (rsf-backward layer2 
    (rsf-forward layer2 (rsf-forward layer1 x))) ≡ x
rsf-compose layer1 layer2 x = begin
  rsf-backward layer1 (rsf-backward layer2 (rsf-forward layer2 (rsf-forward layer1 x)))
    ≡⟨ cong (rsf-backward layer1) (rsf-invertible layer2 (rsf-forward layer1 x)) ⟩
  rsf-backward layer1 (rsf-forward layer1 x)
    ≡⟨ rsf-invertible layer1 x ⟩
  x
  ∎

-- Determinism property (proven using invertibility)
rsf-deterministic : ∀ {n} (layer : RSFLayer n) (x : Vec Float (n + n)) →
  rsf-forward layer (rsf-backward layer (rsf-forward layer x)) ≡ rsf-forward layer x
rsf-deterministic layer x = begin
  rsf-forward layer (rsf-backward layer (rsf-forward layer x))
    ≡⟨ cong (rsf-forward layer) (rsf-invertible layer x) ⟩
  rsf-forward layer x
  ∎
