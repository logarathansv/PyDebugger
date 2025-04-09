import ast
import sys
import traceback
import os
import json
from io import StringIO
import streamlit as st
from contextlib import redirect_stdout, redirect_stderr


class DebugSandbox:
    def __init__(self):
        self.breakpoints = set()
        self.current_line = 0
        self.variables = {}
        self.execution_log = []

    def execute_with_debug(self, code):
        """Execute code with debug capabilities"""
        tree = ast.parse(code)
        self._instrument_code(tree)

        output = StringIO()
        error = StringIO()

        # Add self reference to the execution namespace
        self.variables['self'] = self

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

        return {
            'output': output.getvalue(),
            'error': error.getvalue(),
            'variables': {k: v for k, v in self.variables.items() if k != 'self'},  # Exclude self from output
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
        log_expr = ast.Expr(value=ast.Call(
            func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_log_execution', ctx=ast.Load()),
            args=[ast.Constant(value=self.current_line)],
            keywords=[]
        ))

        # Set location info from original node
        log_expr = ast.copy_location(log_expr, node)

        if isinstance(node, ast.FunctionDef):
            node.body = [log_expr] + node.body
            return node
        else:
            new_node = ast.If(
                test=ast.Constant(value=True),
                body=[log_expr, node],
                orelse=[]
            )
            new_node = ast.copy_location(new_node, node)
            return new_node

    def _log_execution(self, line_num):
        """Log execution details"""
        self.execution_log.append({
            'line': line_num,
            'variables': {k: v for k, v in self.variables.items() if k != 'self'},  # Exclude self from logs
            'call_stack': traceback.extract_stack()[:-2]
        })


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

def show_debug_sandbox():
    st.title("🐞 Python Debugging Sandbox")

    st.markdown("""
        <style>
        .stTextArea textarea {
        width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)

    code = st.text_area("Enter Python code to debug:", height=300, key="debug_code")
    if st.button("Run with Debugging"):
        sandbox = DebugSandbox()
        result = sandbox.execute_with_debug(code)

        st.session_state.debug_result = result
        st.session_state.stored_code = code

    if 'debug_result' in st.session_state:
        result = st.session_state.debug_result
        code = st.session_state.stored_code

        # with col2:
        #     st.markdown("**Execution Trace**")
        #     if result['execution_log']:
        #         for log in result['execution_log']:
        #             st.code(f"Line {log['line']}", language='python')
        #             with st.expander("Variables at this point"):
        #                 st.json(log['variables'])
        #     else:
        #         st.info("No execution steps recorded")

        st.markdown("### Execution Results")
        tab1, tab3 = st.tabs(["Output", "Errors"])

        with tab1:
            st.code(result['output'] or "No output", language='text')

        # with tab2:
        #     st.json(result['variables'])

        with tab3:
            if result['error']:
                # Enhanced error display
                st.error("**Error Details**")
                st.code(result['error'], language='python')
                
                # Improved error parsing
                error_lines = result['error'].strip().split('\n')
                if error_lines:
                    # Get last line which typically contains the error type and message
                    last_line = error_lines[-1]
                    
                    # Extract error type and message
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
                    
                    # Find the line number in the traceback
                    error_line = None
                    for line in error_lines:
                        if "line " in line.lower() and ", in " in line.lower():
                            try:
                                line_part = line.split("line ")[1]
                                line_num = line_part.split(",")[0] if "," in line_part else line_part
                                error_line = int(line_num.strip())
                                # break
                            except (IndexError, ValueError):
                                continue
                    
                    if error_line is not None:
                        st.error(f"**Error Location:** Line {error_line}")
                        st.code(get_line_code(code, error_line), language='python')
            else:
                st.success("No errors detected")

        # if result['execution_log']:
        #     st.markdown("### Step-through Debugger")
        #     if len(result['execution_log']) > 1:
        #         current_step = st.slider(
        #             "Execution step", 
        #             0, 
        #             len(result['execution_log']) - 1, 
        #             0,
        #             key="debug_step"
        #         )
        #     else:
        #         current_step = 0
            
        #     selected_step = result['execution_log'][current_step]
        #     st.markdown(f"**Line {selected_step['line']}**")
        #     st.code(get_line_code(code, selected_step['line']), language='python')
        #     st.markdown("**Variable State**")
        #     st.json(selected_step['variables'])

show_debug_sandbox()
