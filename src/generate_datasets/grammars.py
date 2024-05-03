def get_grammars(args):
    if args.grammar_to_use == '1':
        grammar_string = \
            """  
            S -> '+' S S [0.25]
            S -> '-' S S [0.05]
            S -> '*' S S [0.1]
    
            S -> '**' '6' 'x_0'[0.02]
            S -> '**' '5' 'x_0'[0.03]
            S -> '**' '4' 'x_0'[0.05]
            S -> '**' '3' 'x_0'[0.05]
            S -> '**' '2' 'x_0'[0.05]
            S -> '**' 'x_1' 'x_0'[0.01]
            S -> 'x_0'      [0.1]
            S -> 'x_1'      [0.1]
            S -> 'c'        [0.1]
    
    
            S -> 'sin' Inner_Function [0.03] 
            S -> 'cos' Inner_Function [0.03] 
            S -> 'log' Inner_Function [0.03]  
    
            Inner_Function -> '+' I I [0.3]
            Inner_Function -> '*' I I [0.3]
            Inner_Function ->  I    [0.4]
    
            I -> 'x_0'          [0.2]
            I -> 'x_1'          [0.2]
            I -> 'c'            [0.2]
            I -> '**' '2' 'x_0'     [0.2]
            I -> '**' '2' 'x_1'     [0.2]
            Variable -> 'x_0'[0.5] | 'x_1' [0.5]
               """
    elif args.grammar_to_use == '2':
        grammar_string = \
            """  
            S -> '+' S S [0.15]
            S -> '-' S S [0.1]
            S -> '*' S S [0.15]
            S -> '/' S S [0.1]
            S -> '**' S S     [0.025]
            S -> '**' 2 'x_0'      [0.05]
            S -> '**' 2 'x_1'      [0.05]
            S -> 'x_0'   [0.1]
            S -> 'x_1'   [0.1]
            S -> 'c'     [0.1]
    
    
            S -> 'sin' I      [0.025] 
            S -> 'cos' I      [0.025]
            S -> 'log' I      [0.025]
    
            I -> '+' I I [0.1]
            I -> '*' I I [0.1]
            I -> '/' I I [0.05]
            I ->  'x_0'      [0.25]
            I ->  'x_1'      [0.25]
            I ->  'c'      [0.25]
            Variable -> 'x_0'[0.5] | 'x_1' [0.5]
               """
    elif args.grammar_to_use == 'nguyen':
        grammar_string = \
            """  
            S -> '+' '+' '**' '3'  'x_0' '**'  '2'  'x_0'  'x_0'   [0.08] 
            S -> '+' '+' '+' '**' '4' 'x_0' '**' '3'  'x_0' '**'  '2'  'x_0'  'x_0'   [0.08]
            S -> '+' '+' '+' '+' '**' 5 'x_0' '**' '4' 'x_0' '**' '3'  'x_0' '**'  '2'  'x_0'  'x_0'   [0.08]
            S -> '+' '+' '+' '+' '+' '**' 6 'x_0' '**' 5 'x_0' '**' '4' 'x_0' '**' '3'  'x_0' '**'  '2'  'x_0'  'x_0'   [0.08]
            S ->  '-' '*' 'sin' '**'  '2'  'x_0' 'cos' 'x_0' '1'    [0.08]
            S -> '+' 'sin' 'x_0' 'sin' '+' 'x_0' '**'  '2'  'x_0'   [0.08]
            S -> '+' log '+' 'x_0' '1'  log '+' '**'  '2'  'x_0' '1'    [0.08]
            S -> '**' '0.5' 'x_0'  [0.12]
            S -> '+' 'sin' 'x_0' 'sin' '**'  '2'  'x_1'   [0.08]
            S -> '*'  '2'  '*' 'sin' 'x_0' 'cos' 'x_1'   [0.08]
            S -> '**' 'x_1'  'x_0'[0.08]
            S -> '-' '+'  '-' '**' '4' 'x_0' '**' '3'  'x_0'  '*' '0.5' '**'  '2'  'x_1'  'x_1' [0.08]
            Variable -> 'x_0'[0.5] | 'x_1'  [0.5]  [0.5]
               """

    elif args.grammar_to_use == '4':
        grammar_string = \
            """  
            S -> '+' S S [0.1875]
            S -> '-' S S [0.125]
            S -> '*' S S [0.125]
            S -> '/' S S [0.125]
            S -> 'sin' Inner_Function [0.0625] 
            S -> 'cos' Inner_Function [0.0625] 
            S -> 'log' Inner_Function [0.0625]  
            S -> '**' Exponent Variable [0.0625]| '**' Exponent S [0.0625] 
            S -> Variable [0.0625]| 'c'  [0.0625]

            Exponent -> '6' [0.125]| '5' [0.125]| '4' [0.125]| '3' [0.125]| '2' [0.125]| '0.5' [0.125]| 'x_0' [0.125]| 'x_1' [0.125]            

            Inner_Function -> '+' I I [0.3]
            Inner_Function -> '*' I I [0.3]
            Inner_Function ->  I    [0.4]

            I -> 'x_0' [0.2]|  'x_1'[0.2]| 'c'[0.2]| '**' '2' 'x_0'[0.2]|  '**' '2' 'x_1'[0.2]
            Variable -> 'x_0'[0.5] | 'x_1' [0.5]
               """

    elif args.grammar_to_use == '5':
        grammar_string = \
            """  
            S -> '6' S[0.05]| '5' S[0.05]| '4' S[0.05]| '3' S[0.05]| '2' S[0.05]| '0.5' S[0.05]
            S -> '+' S[0.05]| '-' S[0.05] |  '*' S[0.05]| '/' S[0.05]
            S -> 'sin' S[0.05]| 'cos' S[0.05] |  'log' S[0.05] | '**' S[0.05] 
            S -> S S []   
            S -> 'x_0' [0.1]| 'x_1' [0.1] | 'c' [0.05]

            Variable -> 'x_0'[0.5] | 'x_1' [0.5]
               """
    elif args.grammar_to_use == 'curated_equations':
        grammar_string = \
            """  
            S -> '+' 'c' Variable [0.05]
            S -> '+' 'c' '**' Power Variable [0.05]
            S -> '+' 'c' 'sin' Variable [0.05]
            S -> '+' 'c' 'cos' Variable [0.05]
            S -> '+' 'c' '**' Power Variable [0.05]
            S -> '-' 'c' '*' 'c' '/' 1 '+'  '**' '2' Variable '1' [0.05]
            S -> '/' 'c'  Variable [0.05]
            S -> '/' 'c' '**' Variable 'c'  [0.05]
            S -> '+' 'c' 'ln' Variable [0.05]
            S -> '**' '0.5' '*' 'c' '**' Power Variable [0.05]
            S -> '**' '**' '3' Variable  'c' [0.05]
            S -> '+' 'c'  '**' '-' '0' '**' Power Variable '2'   [0.04]
            S -> '/' '1' '+' '1' '**' Variable 'c'  [0.04]
            S -> '+' 'c' '**' Power Variable [0.04]
            S -> '-' '1' '+' '*' 'c' '**' '3' Variable '+' '*' 'c' '**' '2' Variable  '*' 'c' Variable [0.04]
            S -> '+' 'c' 'sin' '*' '2' Variable [0.04]
            S -> '+' 'c' 'cos' '*' '2' Variable [0.05]
            S ->  '+' '*' 'c' '**' Power Variable '+' '*' 'c' '**' Power Variable  '*' 'c' Variable [0.05]
            S ->  '+' '*' 'c' '**' Power Variable  '+' '*' 'c' '**' Power Variable '+' '*' 'c' '**' Power Variable  '*' 'c' Variable  [0.05]           
            S -> '-' 'c' Variable [0.05]
            S -> '-' 'c' '**' Power Variable [0.05]
            Power -> '0.33' [0.2]| '0.5' [0.2]| '2' [0.2] | '3' [0.2] | '4' [0.2]
            Variable -> 'x_0'[0.5] | 'x_1' [0.5] 
               """

    elif args.grammar_to_use == 'equation_types':
        grammar_string = \
            """  
            S -> '+' 'c' Variable [0.05]
            S -> '+' 'c' '**' Power Variable [0.05]
            S -> '+' 'c' 'sin' Variable [0.05]
            S -> '+' 'c' 'cos' Variable [0.05]
            S -> '+' 'c' '**' Power Variable [0.05]
            S -> '-' 'c' '*' 'c' '/' 1 '+'  '**' '2' Variable '1' [0.05]
            S -> '/' 'c'  Variable [0.05]
            S -> '/' 'c' '**' Variable 'c'  [0.05]
            S -> '+' 'c' 'ln' Variable [0.05]
            S -> '**' '0.5' '*' 'c' '**' Power Variable [0.05]
            S -> '**' '**' '3' Variable  'c' [0.05]
            S -> '+' 'c'  '**' '-' '0' '**' Power Variable '2'   [0.04]
            S -> '/' '1' '+' '1' '**' Variable 'c'  [0.04]
            S -> '+' 'c' '**' Power Variable [0.04]
            S -> '-' '1' '+' '*' 'c' '**' '3' Variable '+' '*' 'c' '**' '2' Variable  '*' 'c' Variable [0.04]
            S -> '+' 'c' 'sin' '*' '2' Variable [0.04]
            S -> '+' 'c' 'cos' '*' '2' Variable [0.05]
            S ->  '+' '*' 'c' '**' Power Variable '+' '*' 'c' '**' Power Variable  '*' 'c' Variable [0.05]
            S ->  '+' '*' 'c' '**' Power Variable  '+' '*' 'c' '**' Power Variable '+' '*' 'c' '**' Power Variable  '*' 'c' Variable  [0.05]           
            S -> '-' 'c' Variable [0.05]
            S -> '-' 'c' '**' Power Variable [0.05]
            Power -> '0.33' [0.2]| '0.5' [0.2]| '2' [0.2] | '3' [0.2] | '4' [0.2]
            Variable -> 'x_0'[0.5] | 'x_1' [0.5] 
               """

    elif args.grammar_to_use == 'curated_equations_old':
        grammar_string = \
            """  
            S -> '+' 'c' 'x_0' [0.025]
            S -> '+' 'c' '**' '2' 'x_0' [0.025]
            S -> '+' 'c' '**' '3' 'x_0' [0.025]
            S -> '+' 'c' 'sin' 'x_0' [0.025]
            S -> '+' 'c' 'cos' 'x_0' [0.025]
            S -> '+' 'c' '**' '0.5' 'x_0' [0.025]
            S -> '-' 'c' '*' 'c' '/' 1 '+'  '**' '2' 'x_0' '1' [0.025]
            S -> '/' 'c'  'x_0' [0.025]
            S -> '/' 'c' '**' 'x_0' 'c'  [0.025]
            S -> '+' 'c' 'ln' 'x_0' [0.025]
            S -> '**' '0.5' '*' 'c' '**' '2' 'x_0' [0.025]
            S -> '**' '**' '3' 'x_0'  'c' [0.025]
            S -> '+' 'c'  '**' '-' '0' '**' '2' 'x_0' '2'   [0.02]
            S -> '/' '1' '+' '1' '**' 'x_0' 'c'  [0.02]
            S -> '+' 'c' '**' '0.33' 'x_0' [0.02]
            S -> '-' '1' '+' '*' 'c' '**' '3' 'x_0' '+' '*' 'c' '**' '2' 'x_0'  '*' 'c' 'x_0' [0.02]
            S -> '+' 'c' 'sin' '*' '2' 'x_0' [0.02]
            S -> '+' 'c' 'cos' '*' '2' 'x_0' [0.02]
            S ->  '+' '*' 'c' '**' '3' 'x_0' '+' '*' 'c' '**' '2' 'x_0'  '*' 'c' 'x_0' [0.02]
            S ->  '+' '*' 'c' '**' '4' 'x_0'  '+' '*' 'c' '**' '3' 'x_0' '+' '*' 'c' '**' '2' 'x_0'  '*' 'c' 'x_0'  [0.02]           
            S -> '-' 'c' x_0 [0.02]
            S -> '-' 'c' '**' '2' 'x_0' [0.02]
            
            S -> '+' 'c' 'x_1' [0.025]
            S -> '+' 'c' '**' '2' 'x_1' [0.025]
            S -> '+' 'c' '**' '3' 'x_1' [0.025]
            S -> '+' 'c' 'sin' 'x_1' [0.025]
            S -> '+' 'c' 'cos' 'x_1' [0.025]
            S -> '+' 'c' '**' '0.5' 'x_1' [0.025]
            S -> '-' 'c' '*' 'c' '/' 1 '+'  '**' '2' 'x_1' '1' [0.025]
            S -> '/' 'c'  'x_1' [0.025]
            S -> '/' 'c' '**' 'x_1' 'c'  [0.025]
            S -> '+' 'c' 'ln' 'x_1' [0.025]
            S -> '**' '0.5' '*' 'c' '**' '2' 'x_1' [0.025]
            S -> '**' '**' '3' 'x_1'  'c' [0.025]
            S -> '+' 'c'  '**' '-' '0' '**' '2' 'x_1' '2'   [0.02]
            S -> '/' '1' '+' '1' '**' 'x_1' 'c'  [0.02]
            S -> '+' 'c' '**' '0.33' 'x_1' [0.02]
            S -> '-' '1' '+' '*' 'c' '**' '3' 'x_1' '+' '*' 'c' '**' '2' 'x_1'  '*' 'c' 'x_1' [0.02]
            S -> '+' 'c' 'sin' '*' '2' 'x_1' [0.02]
            S -> '+' 'c' 'cos' '*' '2' 'x_1' [0.02]
            S ->  '+' '*' 'c' '**' '3' 'x_1' '+' '*' 'c' '**' '2' 'x_1'  '*' 'c' 'x_1' [0.02]
            S ->  '+' '*' 'c' '**' '4' 'x_1'  '+' '*' 'c' '**' '3' 'x_1' '+' '*' 'c' '**' '2' 'x_1'  '*' 'c' 'x_1'  [0.02]           
            S -> '-' 'c' x_1 [0.02]
            S -> '-' 'c' '**' '2' 'x_1' [0.02]

            Variable -> 'x_0'[0.5] | 'x_1' [0.5]
               """

    else:
        raise AssertionError('grammar you want to use does not exist')

    return grammar_string
