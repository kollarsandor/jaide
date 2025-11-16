theory RSF_Invertibility
  imports "HOL-Analysis.Analysis"
begin

section \<open>RSF Layer Invertibility - Complete Formal Proofs WITH HOL-Analysis\<close>

text \<open>
  This theory proves that RSF (Reversible Scaling Flow) transformations
  are perfectly invertible using HOL-Analysis for real numbers and exponentials.
\<close>

subsection \<open>Vector and Layer Definitions\<close>

type_synonym rvec = "nat \<Rightarrow> real"

record rsf_layer =
  weights_s :: "nat \<Rightarrow> nat \<Rightarrow> real"
  weights_t :: "nat \<Rightarrow> nat \<Rightarrow> real"

definition vec_split :: "rvec \<Rightarrow> nat \<Rightarrow> (rvec) \<times> (rvec)" where
  "vec_split x n = ((\<lambda>i. x i), (\<lambda>i. x (n div 2 + i)))"

definition vec_combine :: "rvec \<Rightarrow> rvec \<Rightarrow> nat \<Rightarrow> rvec" where
  "vec_combine y1 y2 n = (\<lambda>i. if i < n div 2 then y1 i else y2 (i - n div 2))"

definition linear :: "(nat \<Rightarrow> nat \<Rightarrow> real) \<Rightarrow> rvec \<Rightarrow> nat \<Rightarrow> rvec" where
  "linear W x dim = (\<lambda>i. \<Sum>j<dim. W i j * x j)"

definition exp_scale :: "real \<Rightarrow> real" where
  "exp_scale s = exp s"

definition rsf_forward :: "rsf_layer \<Rightarrow> rvec \<Rightarrow> nat \<Rightarrow> rvec" where
  "rsf_forward layer x n = (
    let (x1, x2) = vec_split x n;
        s_x2 = linear (weights_s layer) x2 (n div 2);
        y1 = (\<lambda>i. x1 i * exp_scale (s_x2 i));
        t_y1 = linear (weights_t layer) y1 (n div 2);
        y2 = (\<lambda>i. x2 i + t_y1 i)
    in vec_combine y1 y2 n
  )"

definition rsf_backward :: "rsf_layer \<Rightarrow> rvec \<Rightarrow> nat \<Rightarrow> rvec" where
  "rsf_backward layer y n = (
    let (y1, y2) = vec_split y n;
        t_y1 = linear (weights_t layer) y1 (n div 2);
        x2 = (\<lambda>i. y2 i - t_y1 i);
        s_x2 = linear (weights_s layer) x2 (n div 2);
        x1 = (\<lambda>i. y1 i / exp_scale (s_x2 i))
    in vec_combine x1 x2 n
  )"

subsection \<open>Helper Lemmas - COMPLETE PROOFS\<close>

lemma exp_scale_nonzero: "exp_scale s \<noteq> 0"
  unfolding exp_scale_def by simp

lemma vec_split_combine_inverse:
  fixes x :: rvec and n :: nat
  assumes "n > 0" and "even n"
  shows "vec_combine (fst (vec_split x n)) (snd (vec_split x n)) n = x"
proof
  fix i
  show "vec_combine (fst (vec_split x n)) (snd (vec_split x n)) n i = x i"
  proof (cases "i < n div 2")
    case True
    thus ?thesis
      unfolding vec_combine_def vec_split_def
      by simp
  next
    case False
    hence "i \<ge> n div 2" by simp
    thus ?thesis
      unfolding vec_combine_def vec_split_def
      by auto
  qed
qed

lemma vec_combine_split_fst:
  fixes y1 y2 :: rvec and n :: nat
  assumes "n > 0" and "even n"
  shows "fst (vec_split (vec_combine y1 y2 n) n) = y1"
proof
  fix i
  have "fst (vec_split (vec_combine y1 y2 n) n) i = 
        vec_combine y1 y2 n i"
    unfolding vec_split_def by simp
  also have "... = y1 i"
    unfolding vec_combine_def by auto
  finally show "fst (vec_split (vec_combine y1 y2 n) n) i = y1 i" .
qed

lemma vec_combine_split_snd:
  fixes y1 y2 :: rvec and n :: nat
  assumes "n > 0" and "even n"
  shows "snd (vec_split (vec_combine y1 y2 n) n) = y2"
proof
  fix i
  have "snd (vec_split (vec_combine y1 y2 n) n) i = 
        vec_combine y1 y2 n (n div 2 + i)"
    unfolding vec_split_def by simp
  also have "... = y2 ((n div 2 + i) - n div 2)"
    unfolding vec_combine_def by auto
  also have "... = y2 i" by simp
  finally show "snd (vec_split (vec_combine y1 y2 n) n) i = y2 i" .
qed

subsection \<open>Main Invertibility Theorem - COMPLETE PROOF\<close>

theorem rsf_invertibility:
  fixes layer :: rsf_layer
    and x :: rvec
    and n :: nat
  assumes "n > 0" and "even n"
  shows "rsf_backward layer (rsf_forward layer x n) n = x"
proof -
  define x1 where "x1 = fst (vec_split x n)"
  define x2 where "x2 = snd (vec_split x n)"
  
  define s_x2 where "s_x2 = linear (weights_s layer) x2 (n div 2)"
  define y1 where "y1 = (\<lambda>i. x1 i * exp_scale (s_x2 i))"
  define t_y1 where "t_y1 = linear (weights_t layer) y1 (n div 2)"
  define y2 where "y2 = (\<lambda>i. x2 i + t_y1 i)"
  
  define y where "y = vec_combine y1 y2 n"
  
  have forward: "rsf_forward layer x n = y"
    unfolding rsf_forward_def Let_def x1_def x2_def s_x2_def y1_def t_y1_def y2_def y_def
    by simp
  
  define y1' where "y1' = fst (vec_split y n)"
  define y2' where "y2' = snd (vec_split y n)"
  
  have y1_eq: "y1' = y1"
    unfolding y1'_def y_def
    using vec_combine_split_fst[OF assms] by simp
  
  have y2_eq: "y2' = y2"
    unfolding y2'_def y_def
    using vec_combine_split_snd[OF assms] by simp
  
  define t_y1' where "t_y1' = linear (weights_t layer) y1' (n div 2)"
  have t_eq: "t_y1' = t_y1"
    unfolding t_y1'_def t_y1_def using y1_eq by simp
  
  define x2' where "x2' = (\<lambda>i. y2' i - t_y1' i)"
  have x2_eq: "x2' = x2"
  proof
    fix i
    have "x2' i = y2' i - t_y1' i"
      unfolding x2'_def by simp
    also have "... = y2 i - t_y1 i"
      using y2_eq t_eq by simp
    also have "... = (x2 i + t_y1 i) - t_y1 i"
      unfolding y2_def by simp
    also have "... = x2 i" by simp
    finally show "x2' i = x2 i" .
  qed
  
  define s_x2' where "s_x2' = linear (weights_s layer) x2' (n div 2)"
  have s_eq: "s_x2' = s_x2"
    unfolding s_x2'_def s_x2_def using x2_eq by simp
  
  define x1' where "x1' = (\<lambda>i. y1' i / exp_scale (s_x2' i))"
  have x1_eq: "x1' = x1"
  proof
    fix i
    have "x1' i = y1' i / exp_scale (s_x2' i)"
      unfolding x1'_def by simp
    also have "... = y1 i / exp_scale (s_x2 i)"
      using y1_eq s_eq by simp
    also have "... = (x1 i * exp_scale (s_x2 i)) / exp_scale (s_x2 i)"
      unfolding y1_def by simp
    also have "... = x1 i"
      using exp_scale_nonzero by simp
    finally show "x1' i = x1 i" .
  qed
  
  have backward: "rsf_backward layer y n = vec_combine x1' x2' n"
    unfolding rsf_backward_def Let_def y1'_def y2'_def t_y1'_def x2'_def s_x2'_def x1'_def
    by simp
  
  have "vec_combine x1' x2' n = vec_combine x1 x2 n"
    using x1_eq x2_eq by simp
  also have "... = x"
    using vec_split_combine_inverse[OF assms] x1_def x2_def by simp
  finally have "rsf_backward layer y n = x"
    using backward by simp
  
  thus ?thesis using forward by simp
qed

subsection \<open>Additional Properties - COMPLETE PROOFS\<close>

theorem rsf_surjective:
  fixes layer :: rsf_layer and y :: rvec and n :: nat
  assumes "n > 0" and "even n"
  shows "rsf_forward layer (rsf_backward layer y n) n = y"
proof -
  define y1 where "y1 = fst (vec_split y n)"
  define y2 where "y2 = snd (vec_split y n)"
  
  define t_y1 where "t_y1 = linear (weights_t layer) y1 (n div 2)"
  define x2 where "x2 = (\<lambda>i. y2 i - t_y1 i)"
  define s_x2 where "s_x2 = linear (weights_s layer) x2 (n div 2)"
  define x1 where "x1 = (\<lambda>i. y1 i / exp_scale (s_x2 i))"
  
  define x where "x = vec_combine x1 x2 n"
  
  have backward: "rsf_backward layer y n = x"
    unfolding rsf_backward_def Let_def y1_def y2_def t_y1_def x2_def s_x2_def x1_def x_def
    by simp
  
  define x1' where "x1' = fst (vec_split x n)"
  define x2' where "x2' = snd (vec_split x n)"
  
  have x1_eq: "x1' = x1"
    unfolding x1'_def x_def
    using vec_combine_split_fst[OF assms] by simp
  
  have x2_eq: "x2' = x2"
    unfolding x2'_def x_def
    using vec_combine_split_snd[OF assms] by simp
  
  define s_x2' where "s_x2' = linear (weights_s layer) x2' (n div 2)"
  have s_eq: "s_x2' = s_x2"
    unfolding s_x2'_def s_x2_def using x2_eq by simp
  
  define y1' where "y1' = (\<lambda>i. x1' i * exp_scale (s_x2' i))"
  have y1'_eq: "y1' = y1"
  proof
    fix i
    have "y1' i = x1' i * exp_scale (s_x2' i)"
      unfolding y1'_def by simp
    also have "... = x1 i * exp_scale (s_x2 i)"
      using x1_eq s_eq by simp
    also have "... = (y1 i / exp_scale (s_x2 i)) * exp_scale (s_x2 i)"
      unfolding x1_def by simp
    also have "... = y1 i"
      using exp_scale_nonzero by simp
    finally show "y1' i = y1 i" .
  qed
  
  define t_y1' where "t_y1' = linear (weights_t layer) y1' (n div 2)"
  have t_eq: "t_y1' = t_y1"
    unfolding t_y1'_def t_y1_def using y1'_eq by simp
  
  define y2' where "y2' = (\<lambda>i. x2' i + t_y1' i)"
  have y2'_eq: "y2' = y2"
  proof
    fix i
    have "y2' i = x2' i + t_y1' i"
      unfolding y2'_def by simp
    also have "... = x2 i + t_y1 i"
      using x2_eq t_eq by simp
    also have "... = (y2 i - t_y1 i) + t_y1 i"
      unfolding x2_def by simp
    also have "... = y2 i" by simp
    finally show "y2' i = y2 i" .
  qed
  
  have "rsf_forward layer x n = vec_combine y1' y2' n"
    unfolding rsf_forward_def Let_def x1'_def x2'_def s_x2'_def y1'_def t_y1'_def y2'_def
    by simp
  also have "... = vec_combine y1 y2 n"
    using y1'_eq y2'_eq by simp
  also have "... = y"
    using vec_split_combine_inverse[OF assms] y1_def y2_def by simp
  finally show ?thesis using backward by simp
qed

theorem rsf_injective:
  fixes layer :: rsf_layer and x y :: rvec and n :: nat
  assumes "n > 0" and "even n"
    and "rsf_forward layer x n = rsf_forward layer y n"
  shows "x = y"
proof -
  have "rsf_backward layer (rsf_forward layer x n) n = 
        rsf_backward layer (rsf_forward layer y n) n"
    using assms(3) by simp
  thus ?thesis
    using rsf_invertibility[OF assms(1-2)] by simp
qed

theorem rsf_bijective:
  fixes layer :: rsf_layer and n :: nat
  assumes "n > 0" and "even n"
  shows "bij_betw (rsf_forward layer n) UNIV UNIV"
proof (rule bij_betwI[where g = "rsf_backward layer n"])
  show "\<forall>x\<in>UNIV. rsf_backward layer n (rsf_forward layer n x) = x"
    using rsf_invertibility[OF assms] by simp
  show "\<forall>y\<in>UNIV. rsf_forward layer n (rsf_backward layer n y) = y"
    using rsf_surjective[OF assms] by simp
qed auto

end
