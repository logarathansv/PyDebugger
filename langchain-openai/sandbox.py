# import ast
# import sys
# import traceback
# import os
# import json
# from io import StringIO
# import streamlit as st
# from contextlib import redirect_stdout, redirect_stderr


# class DebugSandbox:
#     def __init__(self):
#         self.breakpoints = set()
#         self.current_line = 0
#         self.variables = {}
#         self.execution_log = []

#     def execute_with_debug(self, code):
#         """Execute code with debug capabilities"""
#         tree = ast.parse(code)
#         self._instrument_code(tree)

#         output = StringIO()
#         error = StringIO()

#         # Add self reference to the execution namespace
#         self.variables['self'] = self

#         try:
#             ast.fix_missing_locations(tree)
#             with redirect_stdout(output), redirect_stderr(error):
#                 exec(compile(tree, '<string>', 'exec'), self.variables)
#         except Exception:
#             error.write(traceback.format_exc())
#         finally:
#             # Remove self reference to avoid pollution
#             if 'self' in self.variables:
#                 del self.variables['self']

#         return {
#             'output': output.getvalue(),
#             'error': error.getvalue(),
#             'variables': {k: v for k, v in self.variables.items() if k != 'self'},  # Exclude self from output
#             'execution_log': self.execution_log
#         }

#     def _instrument_code(self, node):
#         """Add debugging instrumentation to AST"""
#         for field, old_value in ast.iter_fields(node):
#             if isinstance(old_value, list):
#                 new_values = []
#                 for value in old_value:
#                     if isinstance(value, ast.stmt):
#                         value = self._instrument_statement(value)
#                     new_values.append(value)
#                 setattr(node, field, new_values)

#     def _instrument_statement(self, node):
#         self.current_line += 1
#         log_expr = ast.Expr(value=ast.Call(
#             func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_log_execution', ctx=ast.Load()),
#             args=[ast.Constant(value=self.current_line)],
#             keywords=[]
#         ))

#         # Set location info from original node
#         log_expr = ast.copy_location(log_expr, node)

#         if isinstance(node, ast.FunctionDef):
#             node.body = [log_expr] + node.body
#             return node
#         else:
#             new_node = ast.If(
#                 test=ast.Constant(value=True),
#                 body=[log_expr, node],
#                 orelse=[]
#             )
#             new_node = ast.copy_location(new_node, node)
#             return new_node

#     def _log_execution(self, line_num):
#         """Log execution details"""
#         self.execution_log.append({
#             'line': line_num,
#             'variables': {k: v for k, v in self.variables.items() if k != 'self'},  # Exclude self from logs
#             'call_stack': traceback.extract_stack()[:-2]
#         })


# def get_line_code(full_code, line_num):
#     """Returns the specific line of code with line number, plus 2 lines of context"""
#     lines = full_code.split('\n')
#     start_line = max(1, line_num - 1)  # Show 1 line before (minimum line 1)
#     end_line = min(len(lines), line_num + 1)  # Show 1 line after
    
#     # Format with line numbers
#     result = []
#     for i in range(start_line, end_line + 1):
#         line_content = lines[i-1]  # Lines are 0-indexed in list
#         prefix = ">>" if i == line_num else f"{i:4}"  # Mark error line
#         result.append(f"{prefix}: {line_content}")
    
#     return '\n'.join(result)

# def show_debug_sandbox():
#     st.title("🐞 Python Debugging Sandbox")

#     st.markdown("""
#         <style>
#         .stTextArea textarea {
#         width: 100% !important;
#     }
#     </style>
#     """, unsafe_allow_html=True)

#     code = st.text_area("Enter Python code to debug:", height=300, key="debug_code")
#     if st.button("Run with Debugging"):
#         sandbox = DebugSandbox()
#         result = sandbox.execute_with_debug(code)

#         st.session_state.debug_result = result
#         st.session_state.stored_code = code

#     if 'debug_result' in st.session_state:
#         result = st.session_state.debug_result
#         code = st.session_state.stored_code

#         # with col2:
#         #     st.markdown("**Execution Trace**")
#         #     if result['execution_log']:
#         #         for log in result['execution_log']:
#         #             st.code(f"Line {log['line']}", language='python')
#         #             with st.expander("Variables at this point"):
#         #                 st.json(log['variables'])
#         #     else:
#         #         st.info("No execution steps recorded")

#         st.markdown("### Execution Results")
#         tab1, tab3 = st.tabs(["Output", "Errors"])

#         with tab1:
#             st.code(result['output'] or "No output", language='text')

#         # with tab2:
#         #     st.json(result['variables'])

#         with tab3:
#             if result['error']:
#                 # Enhanced error display
#                 st.error("**Error Details**")
#                 st.code(result['error'], language='python')
                
#                 # Improved error parsing
#                 error_lines = result['error'].strip().split('\n')
#                 if error_lines:
#                     # Get last line which typically contains the error type and message
#                     last_line = error_lines[-1]
                    
#                     # Extract error type and message
#                     if ':' in last_line:
#                         error_type, error_message = last_line.split(':', 1)
#                         error_type = error_type.strip()
#                         error_message = error_message.strip()
#                     else:
#                         error_type = "Error"
#                         error_message = last_line.strip()
                    
#                     st.markdown("**Debug Summary**")
#                     st.error(f"**Error Type:** `{error_type}`")
#                     st.error(f"**Error Message:** `{error_message}`")
                    
#                     # Find the line number in the traceback
#                     error_line = None
#                     for line in error_lines:
#                         if "line " in line.lower() and ", in " in line.lower():
#                             try:
#                                 line_part = line.split("line ")[1]
#                                 line_num = line_part.split(",")[0] if "," in line_part else line_part
#                                 error_line = int(line_num.strip())
#                                 # break
#                             except (IndexError, ValueError):
#                                 continue
                    
#                     if error_line is not None:
#                         st.error(f"**Error Location:** Line {error_line}")
#                         st.code(get_line_code(code, error_line), language='python')
#             else:
#                 st.success("No errors detected")

#         # if result['execution_log']:
#         #     st.markdown("### Step-through Debugger")
#         #     if len(result['execution_log']) > 1:
#         #         current_step = st.slider(
#         #             "Execution step", 
#         #             0, 
#         #             len(result['execution_log']) - 1, 
#         #             0,
#         #             key="debug_step"
#         #         )
#         #     else:
#         #         current_step = 0
            
#         #     selected_step = result['execution_log'][current_step]
#         #     st.markdown(f"**Line {selected_step['line']}**")
#         #     st.code(get_line_code(code, selected_step['line']), language='python')
#         #     st.markdown("**Variable State**")
#         #     st.json(selected_step['variables'])

# show_debug_sandbox()

# #-------------------------------------------------

import ast
import sys
import traceback
import os
import json
from io import StringIO
import streamlit as st
from contextlib import redirect_stdout, redirect_stderr
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import inspect
import pprint
import time
from streamlit.components.v1 import html

class DebugSandbox:
    def __init__(self):
        self.breakpoints = set()
        self.current_line = 0
        self.variables = {}
        self.execution_log = []
        self.execution_paused = False
        self.breakpoint_hit = False

    def execute_with_debug(self, code, step_mode=False):
        """Execute code with debug capabilities"""
        tree = ast.parse(code)
        self._instrument_code(tree)

        output = StringIO()
        error = StringIO()

        # Add self reference to the execution namespace
        self.variables['self'] = self
        self.variables['__step_mode__'] = step_mode

        try:
            ast.fix_missing_locations(tree)
            with redirect_stdout(output), redirect_stderr(error):
                exec(compile(tree, '<string>', 'exec'), self.variables)
        except Exception:
            error.write(traceback.format_exc())
        finally:
            # Remove self reference to avoid pollution
            if 'self' in self.variables:
                del self.variables['self']
            if '__step_mode__' in self.variables:
                del self.variables['__step_mode__']

        return {
            'output': output.getvalue(),
            'error': error.getvalue(),
            'variables': {k: v for k, v in self.variables.items() if not k.startswith('__')},
            'execution_log': self.execution_log
        }

    def _instrument_code(self, node):
        """Add debugging instrumentation to AST"""
        for field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast.stmt):
                        value = self._instrument_statement(value)
                    new_values.append(value)
                setattr(node, field, new_values)

    def _instrument_statement(self, node):
        self.current_line += 1
        line_num = self.current_line
        
        # Create breakpoint check
        breakpoint_check = ast.If(
            test=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='self', ctx=ast.Load()),
                    attr='_check_breakpoint',
                    ctx=ast.Load()
                ),
                args=[ast.Constant(value=line_num)],
                keywords=[]
            ),
            body=[ast.Pass()],
            orelse=[]
        )
        
        # Create step mode check
        step_check = ast.If(
            test=ast.Name(id='__step_mode__', ctx=ast.Load()),
            body=[
                ast.Expr(value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='self', ctx=ast.Load()),
                        attr='_pause_execution',
                        ctx=ast.Load()
                    ),
                    args=[ast.Constant(value=line_num)],
                    keywords=[]
                ))
            ],
            orelse=[]
        )
        
        # Create log expression
        log_expr = ast.Expr(value=ast.Call(
            func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), 
            attr='_log_execution', 
            ctx=ast.Load()),
            args=[ast.Constant(value=line_num)],
            keywords=[]
        ))

        # Set location info from original node
        log_expr = ast.copy_location(log_expr, node)
        breakpoint_check = ast.copy_location(breakpoint_check, node)
        step_check = ast.copy_location(step_check, node)

        if isinstance(node, ast.FunctionDef):
            node.body = [breakpoint_check, step_check, log_expr] + node.body
            return node
        else:
            new_node = ast.If(
                test=ast.Constant(value=True),
                body=[breakpoint_check, step_check, log_expr, node],
                orelse=[]
            )
            new_node = ast.copy_location(new_node, node)
            return new_node

    def _log_execution(self, line_num):
        """Log execution details"""
        self.execution_log.append({
            'line': line_num,
            'variables': {k: v for k, v in self.variables.items() if not k.startswith('__')},
            'call_stack': traceback.extract_stack()[:-2]
        })

    def _check_breakpoint(self, line_num):
        """Check if execution should pause at this line"""
        if line_num in self.breakpoints:
            self.breakpoint_hit = True
            self.execution_paused = True
            return True
        return False

    def _pause_execution(self, line_num):
        """Pause execution for step-through debugging"""
        self.execution_paused = True
        while self.execution_paused:
            time.sleep(0.1)

    def add_breakpoint(self, line_num):
        """Add a breakpoint at the specified line number"""
        self.breakpoints.add(line_num)

    def remove_breakpoint(self, line_num):
        """Remove breakpoint from the specified line number"""
        if line_num in self.breakpoints:
            self.breakpoints.remove(line_num)

    def continue_execution(self):
        """Resume execution after pause"""
        self.execution_paused = False
        self.breakpoint_hit = False


def format_variable_display(var):
    """Format variables for pretty display"""
    if isinstance(var, (int, float, str, bool, type(None))):
        return var
    try:
        return pprint.pformat(var, indent=2, width=60, depth=3)
    except:
        return str(var)


def get_line_code(full_code, line_num):
    """Returns the specific line of code with line number, plus 2 lines of context"""
    lines = full_code.split('\n')
    start_line = max(1, line_num - 1)  # Show 1 line before (minimum line 1)
    end_line = min(len(lines), line_num + 1)  # Show 1 line after
    
    # Format with line numbers
    result = []
    for i in range(start_line, end_line + 1):
        line_content = lines[i-1]  # Lines are 0-indexed in list
        prefix = ">>" if i == line_num else f"{i:4}"  # Mark error line
        result.append(f"{prefix}: {line_content}")
    
    return '\n'.join(result)


def highlight_code(code):
    """Apply syntax highlighting to code"""
    formatter = HtmlFormatter(style='friendly')
    highlighted = highlight(code, PythonLexer(), formatter)
    return f'<style>{formatter.get_style_defs()}</style>{highlighted}'


def show_debug_sandbox():
    st.title("🐞 Enhanced Python Debugging Sandbox")

    # Custom CSS for better styling
    st.markdown("""
        <style>
        .stTextArea textarea {
            font-family: monospace;
            font-size: 14px;
            line-height: 1.5;
        }
        .debug-container {
            border: 1px solid #eee;
            border-radius: 5px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .variable-display {
            font-family: monospace;
            white-space: pre;
            background-color: #f8f8f8;
            padding: 0.5rem;
            border-radius: 3px;
            max-height: 300px;
            overflow-y: auto;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'debug_result' not in st.session_state:
        st.session_state.debug_result = None
    if 'stored_code' not in st.session_state:
        st.session_state.stored_code = ""
    if 'breakpoints' not in st.session_state:
        st.session_state.breakpoints = set()
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'execution_mode' not in st.session_state:
        st.session_state.execution_mode = "Run"  # Default to "Run"

    # Code editor section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        code = st.text_area(
            "Enter Python code to debug:", 
            height=400,
            value=st.session_state.get('stored_code', ''),
            key="debug_code"
        )
    
    with col2:
        st.markdown("**Debug Controls**")
        execution_mode = st.radio(
            "Execution mode:",
            ["Run", "Step-through"],
            index=0 if st.session_state.execution_mode == "Run" else 1,
            key="exec_mode_radio"
        )
        
        if st.button("Run Code"):
            sandbox = DebugSandbox()
            for bp in st.session_state.breakpoints:
                sandbox.add_breakpoint(bp)
                
            result = sandbox.execute_with_debug(
                code, 
                step_mode=(execution_mode == "Step-through")
            )
            
            st.session_state.debug_result = result
            st.session_state.stored_code = code
            st.session_state.current_step = 0
            st.session_state.execution_mode = execution_mode
            st.rerun()

        if st.session_state.debug_result and st.session_state.execution_mode == "Step-through":
            if st.button("Continue"):
                sandbox = DebugSandbox()
                sandbox.continue_execution()
                st.rerun()

    # Breakpoint management
    st.markdown("### Breakpoints")
    if code:
        lines = code.split('\n')
        selected_lines = st.multiselect(
            "Select lines to add breakpoints:",
            options=[(i+1, lines[i][:50] + ("..." if len(lines[i]) > 50 else "")) 
                     for i in range(len(lines))],
            format_func=lambda x: f"Line {x[0]}: {x[1]}",
            key="breakpoint_selector"
        )
        
        if selected_lines:
            st.session_state.breakpoints = {line[0] for line in selected_lines}
            st.info(f"Breakpoints set at lines: {', '.join(str(line[0]) for line in selected_lines)}")

    # Display results
    if st.session_state.debug_result:
        result = st.session_state.debug_result
        code = st.session_state.stored_code

        st.markdown("### Execution Results")
        tab1, tab2, tab3 = st.tabs(["Output", "Variables", "Errors"])

        with tab1:
            st.code(result['output'] or "No output", language='text')

        with tab2:
            if result['variables']:
                st.markdown("**Final Variable States**")
                for name, value in result['variables'].items():
                    with st.expander(f"`{name}`"):
                        st.markdown(f"```python\n{format_variable_display(value)}\n```", unsafe_allow_html=True)
            else:
                st.info("No variables defined")

        with tab3:
            if result['error']:
                st.error("**Error Details**")
                st.code(result['error'], language='python')
                
                error_lines = result['error'].strip().split('\n')
                if error_lines:
                    last_line = error_lines[-1]
                    
                    if ':' in last_line:
                        error_type, error_message = last_line.split(':', 1)
                        error_type = error_type.strip()
                        error_message = error_message.strip()
                    else:
                        error_type = "Error"
                        error_message = last_line.strip()
                    
                    st.markdown("**Debug Summary**")
                    st.error(f"**Error Type:** `{error_type}`")
                    st.error(f"**Error Message:** `{error_message}`")
                    
                    error_line = None
                    for line in error_lines:
                        if "line " in line.lower() and ", in " in line.lower():
                            try:
                                line_part = line.split("line ")[1]
                                line_num = line_part.split(",")[0] if "," in line_part else line_part
                                error_line = int(line_num.strip())
                            except (IndexError, ValueError):
                                continue
                    
                    if error_line is not None:
                        st.error(f"**Error Location:** Line {error_line}")
                        st.code(get_line_code(code, error_line), language='python')
            else:
                st.success("No errors detected")

        # Execution trace and step-through debugger
        if result.get('execution_log'):
            st.markdown("### Execution Trace")
            
            if st.session_state.execution_mode == "Step-through" or len(result['execution_log']) > 1:
                current_step = st.slider(
                    "Execution step", 
                    0, 
                    len(result['execution_log']) - 1, 
                    st.session_state.current_step,
                    key="debug_step_slider"
                )
                st.session_state.current_step = current_step
            else:
                current_step = 0
            
            selected_step = result['execution_log'][current_step]
            
            st.markdown(f"**Executing Line {selected_step['line']}**")
            st.markdown(highlight_code(get_line_code(code, selected_step['line'])), unsafe_allow_html=True)
            
            st.markdown("**Variable State at This Point**")
            if selected_step['variables']:
                for name, value in selected_step['variables'].items():
                    with st.expander(f"`{name}`"):
                        st.markdown(f"```python\n{format_variable_display(value)}\n```", unsafe_allow_html=True)
            else:
                st.info("No variables defined at this point")

# Run the app
if __name__ == "__main__":
    show_debug_sandbox()