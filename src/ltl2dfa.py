# output LTL syntax is from http://ltlf2dfa.diag.uniroma1.it/ltlf_syntax
# with the addition of parenthesis for grouping
from FiniteStateMachine import MooreMachine


def ltl_ast2str(ast) -> str:
    if not isinstance(ast, tuple):
        assert isinstance(ast, str)
        return ast
    op, *args = ast
    # One case for each type of sampler in src/ltl_samplers.py
    if op == 'or':
        return f"({ltl_ast2str(args[0])}) | ({ltl_ast2str(args[1])})"
    elif op == 'until':
        return f"({ltl_ast2str(args[0])}) U ({ltl_ast2str(args[1])})"
    elif op == 'and':
        return f"({ltl_ast2str(args[0])}) & ({ltl_ast2str(args[1])})"
    elif op == 'not':
        return f"!({ltl_ast2str(args[0])})"
    elif op == 'eventually':
        return f"F ({ltl_ast2str(args[0])})"


def ltl2dfa(ltl_ast, symbols, name='PLACEHOLDER'):
    ltl = ltl_ast2str(ltl_ast)
    # print(f'converting {ltl} to DFA...')
    return MooreMachine(
        ltl,
        len(symbols),
        name,
        dictionary_symbols=symbols
    )


def test():
    from ltl_samplers import DefaultSampler
    from pprint import pprint
    for _ in range(5):
        ast = DefaultSampler(['a', 'b', 'c', 'd', 'e', 'f']).sample()
        print('ast:')
        pprint(ast)
        print('ltl:')
        print(ltl_ast2str(ast))

if __name__ == '__main__':
    test()

