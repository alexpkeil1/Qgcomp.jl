# utility.jl: utility functions for working with Julia objects

degree_term(t::AbstractTerm, d::ConstantTerm{Int64}) = FunctionTerm(^, [t,d], :($t^$d))

#=
degreebuilder("mixture", 4)
=#
function degreebuilder(var, degree)
    f = ConstantTerm(1)
    for d = 1:degree
        f += d > 1 ? degree_term(Term(Symbol("$var")), ConstantTerm(d)) : Term(Symbol("$var"))
    end
    f
end


function subterm!(rhs, idx, t, expnms)
    if typeof(t) <: Term
        if rhs[idx].sym == :mixture
            rhs[idx] = Term(Symbol(expnms[1]))
        end
    elseif typeof(t) <: FunctionTerm
        if t.args[1].sym == :mixture
            rhs[idx].args[1] = Term(Symbol(expnms[1]))
        end
        if t.exorig.args[2] == :mixture
            rhs[idx].exorig.args[2] = Symbol(expnms[1])
        end
    end
    nothing
end


function subformula(f, expnms)
    rhs = deepcopy([r for r in f.rhs])
    for (idx, r) in enumerate(rhs)
        subterm!(rhs, idx, r, expnms)
    end
    FormulaTerm(f.lhs, Tuple(rhs))
end

function sublhs(f, symb)
    lhs = typeof(f.lhs)(symb)
    FormulaTerm(lhs, f.rhs)
end


function sublhs_surv(f, symb)
    lhs = f.lhs
    lhs.args[end] = typeof(lhs.args[end])(symb)
    lhs.exorig.args[end]  = typeof(lhs.exorig.args[end])(symb)
    FormulaTerm(lhs, f.rhs)
end

function sublhs_surv(f, symbout, simbd)
    lhs = f.lhs
    lhs.args[end-1] = typeof(lhs.args[end])(symbout)
    lhs.args[end] = typeof(lhs.args[end])(simbd)
    lhs.exorig.args[end-1]  = typeof(lhs.exorig.args[end-1])(symbout)
    lhs.exorig.args[end]  = typeof(lhs.exorig.args[end])(simbd)
    FormulaTerm(lhs, f.rhs)
end
