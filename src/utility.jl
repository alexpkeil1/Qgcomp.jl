# utility.jl: utility functions for working with Julia objects

degree_term(t::AbstractTerm, d::ConstantTerm{Int64}) = FunctionTerm(^, [t, d], :($t^$d))

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

#=
expnms = ["x1", "x3"]
checklinearadditive(@formula(y~x1+x2+x3+z1+z1^2).rhs, expnms)
checklinearadditive(@formula(y~x1+x2+x1*x2+x3+z1+z1^2).rhs, expnms)
checklinearadditive(@formula(y~x1+x2 + x1^2+x3+z1+z1^2).rhs, expnms)
checklinearadditive(@formula(y~x1+x2 + sqrt(x1)+x3+z1+z1^2).rhs, expnms)
checklinearadditive(@formula(y~x1+x2 + sqrt(x2)+x3+z1+z1^2).rhs, expnms)
=#

function getsyms(term::T) where {T<:InteractionTerm}
    [t.sym for t in terms(term)]
end

function checklinearadditive(rhs, expnms)
    nexp = length(unique(expnms))
    exterm_count = [0]
    stillvalid = [true]
    for t in rhs
        if hasfield(typeof(t), :sym)
            exterm_count .+= Int(t.sym ∈ expnms)
            exterm_count[1] > nexp && return(false)
        elseif typeof(t)<:InteractionTerm
            syms = getsyms(t)
            if any([any(syms .== Symbol(ex)) for ex in expnms]) 
                stillvalid .= false
            end
            !stillvalid[1] && return(false)        
        elseif hasfield(typeof(t), :terms)
        elseif typeof(t)<:FunctionTerm
            argy = t.exorig.args
            if any([any(argy .== Symbol(ex)) for ex in expnms]) 
                stillvalid .= false
            end
            !stillvalid[1] && return(false)
        end
    end
    true
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
    rhs = length(terms(f.rhs))==1 ? deepcopy(terms(f.rhs)) : [r for r in f.rhs]
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
    lhs.exorig.args[end] = typeof(lhs.exorig.args[end])(symb)
    FormulaTerm(lhs, f.rhs)
end

function sublhs_surv(f, symbout, simbd)
    lhs = f.lhs
    lhs.args[end-1] = typeof(lhs.args[end])(symbout)
    lhs.args[end] = typeof(lhs.args[end])(simbd)
    lhs.exorig.args[end-1] = typeof(lhs.exorig.args[end-1])(symbout)
    lhs.exorig.args[end] = typeof(lhs.exorig.args[end])(simbd)
    FormulaTerm(lhs, f.rhs)
end


function msmcoxcheck(y::Y) where {Y<:LSurvival.AbstractLSurvivalResp}
    msg = "Person-period data and late-entry is not permitted in this function. 
    Time-to-event outcomes that need to be specified as

    `Surv(<enter time>, <exit time>, <event indicator>)`

    are not allowed]. If you have no late entry, you can modify the outcome
    specification to  

    `Surv(<exit time>, <event indicator>)`

    Note: If you have late entry and omit <enter time> to avoid this error message, 
    you will have bias in your estimates.
    "
    @assert allequal(y.enter)
end
