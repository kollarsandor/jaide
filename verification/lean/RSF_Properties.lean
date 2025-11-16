-- JAIDE v40 Lean Formal Verification
-- Complete proofs that RSF (Reversible Scaling Flow) layers are invertible
-- WITH Mathlib dependencies - full formal proofs using Real numbers and tactics

import Mathlib.Data.Real.Basic
import Mathlib.Tactic
import Mathlib.Analysis.SpecialFunctions.Exp

namespace JAIDE.RSF

open Real

-- Define RSF layer structure with Real-valued weights  
structure RSFLayer (n : Nat) where
  weights_s : Fin n → Fin n → ℝ
  weights_t : Fin n → Fin n → ℝ

-- Vector type using Real numbers
def Vec (n : Nat) := Fin n → ℝ

-- Split vector into two halves (for even n)
def vec_split {n : Nat} (x : Vec n) (h : n % 2 = 0) : Vec (n / 2) × Vec (n / 2) :=
  (fun i => x ⟨i.val, Nat.lt_of_lt_of_le i.isLt (Nat.div_le_self n 2)⟩,
   fun i => x ⟨(n / 2) + i.val, by
     have h1 : n / 2 + i.val < n / 2 + n / 2 := Nat.add_lt_add_left i.isLt (n / 2)
     have h2 : n / 2 + n / 2 ≤ n := by
       cases n with
       | zero => simp
       | succ n' => 
         have : 2 * (n'.succ / 2) ≤ n'.succ := Nat.mul_div_le n'.succ 2
         omega
     exact Nat.lt_of_lt_of_le h1 h2⟩)

-- Combine two half-vectors into full vector
def vec_combine {n : Nat} (y1 y2 : Vec (n / 2)) (h : n % 2 = 0) : Vec n :=
  fun i => 
    if hi : i.val < n / 2 then 
      y1 ⟨i.val, by
        have : n / 2 > 0 := by
          cases n with
          | zero => contradiction
          | succ n' => simp; omega
        omega⟩
    else 
      y2 ⟨i.val - n / 2, by omega⟩

-- Linear transformation with weights
def linear {m : Nat} (W : Fin m → Fin m → ℝ) (x : Vec m) : Vec m :=
  fun i => 
    let rec sum_aux (k : Nat) (acc : ℝ) : ℝ :=
      if h : k < m then
        sum_aux (k + 1) (acc + W i ⟨k, h⟩ * x ⟨k, h⟩)
      else
        acc
      termination_by m - k
    sum_aux 0 0

-- Exponential scaling using Real.exp
noncomputable def exp_scale (s : ℝ) : ℝ := Real.exp s

-- RSF forward pass
noncomputable def rsf_forward {n : Nat} (layer : RSFLayer (n / 2)) (x : Vec n) (h : n % 2 = 0) : Vec n :=
  let (x1, x2) := vec_split x h
  let s_x2 := linear layer.weights_s x2
  let y1 := fun i => x1 i * exp_scale (s_x2 i)
  let t_y1 := linear layer.weights_t y1
  let y2 := fun i => x2 i + t_y1 i
  vec_combine y1 y2 h

-- RSF backward pass
noncomputable def rsf_backward {n : Nat} (layer : RSFLayer (n / 2)) (y : Vec n) (h : n % 2 = 0) : Vec n :=
  let (y1, y2) := vec_split y h
  let t_y1 := linear layer.weights_t y1
  let x2 := fun i => y2 i - t_y1 i
  let s_x2 := linear layer.weights_s x2
  let x1 := fun i => y1 i / exp_scale (s_x2 i)
  vec_combine x1 x2 h

-- Helper lemma: split and combine are inverse operations
theorem split_combine_inverse {n : Nat} (x : Vec n) (h : n % 2 = 0) :
    let (x1, x2) := vec_split x h
    vec_combine x1 x2 h = x := by
  ext i
  unfold vec_combine vec_split
  simp
  split
  · rfl
  · simp [*]

-- Helper lemma: combine then split gives back first component
theorem combine_split_left {n : Nat} (y1 y2 : Vec (n / 2)) (h : n % 2 = 0) :
    (vec_split (vec_combine y1 y2 h) h).1 = y1 := by
  ext i
  unfold vec_split vec_combine
  simp

-- Helper lemma: combine then split gives back second component
theorem combine_split_right {n : Nat} (y1 y2 : Vec (n / 2)) (h : n % 2 = 0) :
    (vec_split (vec_combine y1 y2 h) h).2 = y2 := by
  ext i
  unfold vec_split vec_combine
  simp

-- Main invertibility theorem: backward ∘ forward = id
theorem rsf_invertible {n : Nat} (layer : RSFLayer (n / 2)) (x : Vec n) (h : n % 2 = 0) :
    rsf_backward layer (rsf_forward layer x h) h = x := by
  unfold rsf_forward rsf_backward
  simp only []
  ext i
  unfold vec_combine vec_split
  simp
  split
  · -- First half case
    have exp_ne_zero : ∀ s : ℝ, exp_scale s ≠ 0 := by
      intro s
      unfold exp_scale
      exact Real.exp_pos s |>.ne'
    rw [div_mul_cancel]
    exact exp_ne_zero _
  · -- Second half case
    ring

-- Forward ∘ backward = id (surjectivity)
theorem rsf_surjective {n : Nat} (layer : RSFLayer (n / 2)) (y : Vec n) (h : n % 2 = 0) :
    rsf_forward layer (rsf_backward layer y h) h = y := by
  unfold rsf_forward rsf_backward
  simp only []
  ext i
  unfold vec_combine vec_split
  simp
  split
  · -- First half case
    have exp_ne_zero : ∀ s : ℝ, exp_scale s ≠ 0 := by
      intro s
      unfold exp_scale
      exact Real.exp_pos s |>.ne'
    rw [mul_div_cancel₀]
    exact exp_ne_zero _
  · -- Second half case
    ring

-- Injectivity: RSF forward is injective
theorem rsf_injective {n : Nat} (layer : RSFLayer (n / 2)) (x y : Vec n) (h : n % 2 = 0) :
    rsf_forward layer x h = rsf_forward layer y h → x = y := by
  intro heq
  have : rsf_backward layer (rsf_forward layer x h) h = 
         rsf_backward layer (rsf_forward layer y h) h := by rw [heq]
  rw [rsf_invertible, rsf_invertible] at this
  exact this

-- Composition property
theorem rsf_compose_invertible {n : Nat} (layer1 layer2 : RSFLayer (n / 2)) 
    (x : Vec n) (h : n % 2 = 0) :
    rsf_backward layer1 (rsf_backward layer2 
      (rsf_forward layer2 (rsf_forward layer1 x h) h) h) h = x := by
  rw [rsf_invertible, rsf_invertible]

-- Determinism
theorem rsf_deterministic {n : Nat} (layer : RSFLayer (n / 2)) (x : Vec n) (h : n % 2 = 0) :
    rsf_forward layer (rsf_backward layer (rsf_forward layer x h) h) h = 
    rsf_forward layer x h := by
  rw [rsf_surjective]

-- Bijectivity
theorem rsf_bijective {n : Nat} (layer : RSFLayer (n / 2)) (h : n % 2 = 0) :
    (∀ x y, rsf_forward layer x h = rsf_forward layer y h → x = y) ∧
    (∀ y, ∃ x, rsf_forward layer x h = y) := by
  constructor
  · exact fun x y => rsf_injective layer x y h
  · intro y
    use rsf_backward layer y h
    exact rsf_surjective layer y h

end JAIDE.RSF
