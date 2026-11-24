### A Pluto.jl notebook ###
# v1.0.3

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

# ╔═╡ de9f88b2-5506-49ed-9bae-93edeccb8da6
using Optimization , OptimizationOptimJL

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

# ╔═╡ 36b406e3-c12c-405c-aae2-8248d3a731a3
md"""

### Ограничения

#### Ограничения на параметры состояния системы 

Это те ограничения которые могут быть использованы для формулирования начальных условия ОДУ. Траектория , которая лежит в границах допустимого диапазона называется допустимой траекторией (admissible trajectory)

Например, начальная скорость и координата машины. 

``\vec{x}(t) = \begin{pmatrix}  0 \\  0 \end{pmatrix}``


Кроме того это могут быть ограничения на эволюцию  системы во времени. 

Например, необходимо чтобы машина оказалась в  нужной точке в момент времени ``t_f``, при этом надо, чтобы она имела нулевую сокрость:

``\vec{x}(t_f) = \begin{pmatrix}  L \\  0 \end{pmatrix}``

Условия на вектор состояния также могут быть в виде неравенств:

``\dot{x}(t) ≥ 0``  - требуем, чтобы скорость была больше нуля

#### Ограничения на управляющее воздействие 

Ограничения могут накладываться и на управляющее воздействие, управляющее воздействие , лежащее в границах допустимой области называте допустимым управляющим воздействием (admissible control)

например, так как мы ограничены мощностью двигателя и тормозов, вектор управляющего воздействия  тоже будет ограничен.

``\begin{pmatrix}  0 \\  0 \end{pmatrix} ≤ \vec{u} ≤ \begin{pmatrix}  M_1 \\  M_2 \end{pmatrix}``


"""

# ╔═╡ 39934f85-7e47-460e-a1ae-270f3eedb265
md"""

### Целевая функция задачи оптимального контроля 

Общая форма целевой функции задачи оптимального управления :

*``J = h(\vec{x}(t_s) , t_f) + \int_{t_0}^{t_f}{g(\vec{x} , \vec{u}(t) , t)}dt``*

``h``  и ``g`` - это некоторые скалярные функции

Ну и задача оптимального контроля формулируется как :

Найти такое допустимое управляющее воздействие, которое дает допустимую траекторий состояния, которая минимизирует (максимизирует) целевую функцию.  

"""

# ╔═╡ 333a44c7-188f-44a6-940c-689113f1246d
function get_u(p)
	V_cur = f_V(p[1:3] , p[4:end])
end

# ╔═╡ d1021da7-dcbe-4b18-9d06-f264c0ea866d
function single_ode_solve(p )
	V_cur = get_u(p)
	#f(x, p, t)
	p = (A,B,V_cur)
	x0 = SVector{2}([0.0 , 0.0])
	tspan = (0.0, 1.0)
	prob = ODEProblem(f, x0, tspan ,p)
	sol = solve(prob, Tsit5(), reltol = 1e-3, abstol = 1e-3)
	return sol
end

# ╔═╡ f829c1eb-466a-4e85-b49b-6c9ecc6af5b1
function single_try(p , x_end)
	sol = single_ode_solve(p)
	return sum((x_end  .- sol(1.0)).^2)
end

# ╔═╡ 6f5ec9e6-455e-4f31-a682-6c154d4c8936
probl = OptimizationProblem(single_try, [1.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0] , [0.3 , 0.0])

# ╔═╡ 8a76a6c0-dc1a-44a5-931e-cc5b845b37e4
xres = solve(probl , NelderMead())

# ╔═╡ c893ebc9-224c-4e47-aa60-66faa2fc69ff
sol_res = single_ode_solve(xres )

# ╔═╡ b08c7d2c-0a67-4834-bc56-50d5a3ada088
plot(sol_res)

# ╔═╡ a3358d9f-163d-4b53-b476-d23f1613210d
u_res = get_u(xres)

# ╔═╡ 6ddb1002-8ee4-4d56-82d1-81b8034e761f
t_plot= range(0,1,100)

# ╔═╡ 5ef28230-3763-4866-90c0-8713b1a362f3
begin 
	Umat = zeros((length(t_plot) , 2))
	for i in 1:length(t_plot)

		Umat[i , :] .= u_res(t_plot[i])
	end
end

# ╔═╡ cd5ad69a-4326-4c7f-914b-216da69647c2
plot(t_plot ,Umat )

# ╔═╡ 4db45248-93f8-46fa-82f4-52a8ade62a49
begin 

function cost_function(p, x_end, x_start, t_start, t_end)
    V_cur = get_u(p, t_start, t_end) # полиномы на [t_start, t_end]
    prob = ODEProblem(f, x_start, (t_start, t_end), (A, B, V_cur))
    sol = solve(prob, Tsit5(), reltol = 1e-4, abstol = 1e-4)
    
    #  (вес 100)
    terminal_cost = 100.0 * sum((x_end .- sol(t_end)).^2)
    
    # энергия ∫ (u(t)^2 + v(t)^2) dt по сетке из 50 точек
    ts = range(t_start, t_end, length=50)
    energy_cost = sum(sum(abs2, V_cur(t)) for t in ts) * (t_end - t_start) / 50.0
    
    return terminal_cost + 0.1 * energy_cost
end


lb = zeros(6)
ub = ones(6)

end

# ╔═╡ Cell order:
# ╠═c239e2e0-e6a2-11f1-a170-cda53aa43fff
# ╟─cdea6217-159d-47aa-9a8d-58d5df529b49
# ╟─5294b776-edd7-4ce9-b1af-0457a41a2467
# ╟─82528482-867d-488a-9d58-2dd9c5911f5c
# ╟─e40581bf-5a53-412a-a381-d8e5241def9a
# ╟─c23faf42-97cf-4ac9-b741-aae1527cfb6e
# ╟─2dd7f9c2-5f7e-4b9e-9a3d-b76c545f1e40
# ╠═15182ca5-25b6-4b5f-823f-3757e7af301c
# ╟─c43b5852-3f6a-49a9-92a1-e56f255098da
# ╠═c185e898-aa6a-4827-9d8b-749bf24cfdc5
# ╠═cce943ae-0fd0-463e-9d02-9a7ecdffd260
# ╠═df1250b6-1c30-4f74-b89b-8edba586112e
# ╟─36b406e3-c12c-405c-aae2-8248d3a731a3
# ╟─39934f85-7e47-460e-a1ae-270f3eedb265
# ╠═de9f88b2-5506-49ed-9bae-93edeccb8da6
# ╠═f829c1eb-466a-4e85-b49b-6c9ecc6af5b1
# ╠═333a44c7-188f-44a6-940c-689113f1246d
# ╠═d1021da7-dcbe-4b18-9d06-f264c0ea866d
# ╠═6f5ec9e6-455e-4f31-a682-6c154d4c8936
# ╠═8a76a6c0-dc1a-44a5-931e-cc5b845b37e4
# ╠═c893ebc9-224c-4e47-aa60-66faa2fc69ff
# ╠═b08c7d2c-0a67-4834-bc56-50d5a3ada088
# ╠═a3358d9f-163d-4b53-b476-d23f1613210d
# ╠═6ddb1002-8ee4-4d56-82d1-81b8034e761f
# ╠═5ef28230-3763-4866-90c0-8713b1a362f3
# ╠═cd5ad69a-4326-4c7f-914b-216da69647c2
# ╠═4db45248-93f8-46fa-82f4-52a8ade62a49
