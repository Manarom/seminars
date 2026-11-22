### A Pluto.jl notebook ###
# v1.1.0

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ c239e2e0-e6a2-11f1-a170-cda53aa43fff
begin 
	topic_folder = abspath(joinpath(@__DIR__ , ".."  ))
	import Pkg
	Pkg.activate(topic_folder)
	using Plots 
	using PlutoUI 
	using OrdinaryDiffEq
	using ScaledPolynomials
	using StaticArrays
	using BenchmarkTools
end

# ╔═╡ cdea6217-159d-47aa-9a8d-58d5df529b49
BP = ScaledPolynomials.BernsteinSymPoly{3};

# ╔═╡ 5294b776-edd7-4ce9-b1af-0457a41a2467
md"""

#### Пример задачи управления 

``x(t)``  - координата 

``\dot{x}(t)`` - скорость движения 

Движением автомобиля мы можем управлять при помощи двух сигналов 

``u(t)`` - ускорение (газ)

``v(t)`` - замедление (торможение)

Уравнение движения: 

``\frac{d}{dt} \dot{x}(t) = u(t) + v(t)`` 				(1)
"""

# ╔═╡ 82528482-867d-488a-9d58-2dd9c5911f5c
md"""

Перепишем уравнение движения в матричной форме:

``\vec{u} = \begin{pmatrix} u(t) \\ v(t) \end{pmatrix}`` - вектор **управляющего воздействия** 


``\vec{x} = \begin{pmatrix} x(t) \\ \dot{x}(t)  \end{pmatrix}`` - вектор **состояния системы** 

"""

# ╔═╡ e40581bf-5a53-412a-a381-d8e5241def9a
md"""
	
Уравнение движения автомобиля можно записать в виде системы обыкновенных дифференциальных уранений :

``\frac{d\vec{x}}{dt} = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix} \vec{x} + 
\begin{pmatrix} 0 & 0 \\ 1 & 1 \end{pmatrix} \vec{u}``

* Мы тут дополнили уравнение (1) тривиальным уравнением: ``\frac{d}{dt}x = \dot{x}(t)``(верхняя строка матричного уравнения выше)

Также будем считать, что мы наблюдаем за объектом на интервале времени:

``t \in [0 , tₛ]``
"""

# ╔═╡ c23faf42-97cf-4ac9-b741-aae1527cfb6e
md""" 

Полная история зависимости ``\vec{x}(t)`` называют траекторией состояния, а ``\vec{v}(t)`` - историей управления или просто управлением

Ну и прежде, чем решать дифур нужны, конечно начальные условия, пусть это будет состояние покоя ``x(0)=0   , \dot{x}(0)=0``
"""

# ╔═╡ 2dd7f9c2-5f7e-4b9e-9a3d-b76c545f1e40
@bind u_params PlutoUI.combine() do Child
	md"""
	Управление (ускорение)
	
	- **u1:** $(Child(Slider(0:0.01:1, default=1.0, show_value=true)))
	- **u2:** $(Child(Slider(0:0.01:1, default=0.0, show_value=true)))
	- **u3:** $(Child(Slider(0:0.01:1, default=0.0, show_value=true)))
	"""
end

# ╔═╡ c43b5852-3f6a-49a9-92a1-e56f255098da
@bind v_params PlutoUI.combine() do Child
	md"""
	Управление (торможение)
	
	- **v1:** $(Child(Slider(0:0.01:1, default=1.0, show_value=true)))
	- **v2:** $(Child(Slider(0:0.01:1, default=0.0, show_value=true)))
	- **v3:** $(Child(Slider(0:0.01:1, default=0.0, show_value=true)))
	"""
end

# ╔═╡ c185e898-aa6a-4827-9d8b-749bf24cfdc5

function f_V(u_params , v_params)  
	u = ScaledPolynomials.ScaledPolynomial(BP , [u_params...] , 0.0 , 1.0)
	v = ScaledPolynomials.ScaledPolynomial(BP , [v_params...] , 0.0 , 1.0)
	t->SVector((u(t) , v(t))) 
end

# ╔═╡ cce943ae-0fd0-463e-9d02-9a7ecdffd260
V = f_V(u_params , v_params)

# ╔═╡ df1250b6-1c30-4f74-b89b-8edba586112e
begin 
	A = SMatrix{2,2}([0.0  1.0; 0.0  0.0])
	B = SMatrix{2,2}([0.0  0.0; 1.0  -1.0])
	f(x, p, t) = begin 
		A,B,V = p
		return A * x + B * V(t)
	end
end

# ╔═╡ 15182ca5-25b6-4b5f-823f-3757e7af301c
begin 
	p = (A,B,V)
	x0 = SVector{2}([0.0 , 0.0])
	tspan = (0.0, 1.0)
	prob = ODEProblem(f, x0, tspan ,p)
	sol = solve(prob, Tsit5(), reltol = 1e-3, abstol = 1e-3)
	pl = plot(sol  , title = "Траектория состояния")
end

# ╔═╡ Cell order:
# ╠═c239e2e0-e6a2-11f1-a170-cda53aa43fff
# ╟─cdea6217-159d-47aa-9a8d-58d5df529b49
# ╟─5294b776-edd7-4ce9-b1af-0457a41a2467
# ╟─82528482-867d-488a-9d58-2dd9c5911f5c
# ╟─e40581bf-5a53-412a-a381-d8e5241def9a
# ╟─c23faf42-97cf-4ac9-b741-aae1527cfb6e
# ╟─2dd7f9c2-5f7e-4b9e-9a3d-b76c545f1e40
# ╟─15182ca5-25b6-4b5f-823f-3757e7af301c
# ╟─c43b5852-3f6a-49a9-92a1-e56f255098da
# ╟─c185e898-aa6a-4827-9d8b-749bf24cfdc5
# ╟─cce943ae-0fd0-463e-9d02-9a7ecdffd260
# ╟─df1250b6-1c30-4f74-b89b-8edba586112e
