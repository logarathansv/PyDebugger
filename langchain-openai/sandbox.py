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
    lines = full_code.split('\n')
    return '\n'.join(lines[max(0, line_num - 2):line_num + 1])


def show_debug_sandbox():
    st.title("🐞 Python Debugging Sandbox")

    col1, col2 = st.columns([3, 2])

    with col1:
        code = st.text_area("Enter Python code to debug:", height=300, key="debug_code")

        if st.button("Run with Debugging"):
            sandbox = DebugSandbox()
            result = sandbox.execute_with_debug(code)

            st.session_state.debug_result = result
            st.session_state.stored_code = code  # ✅ no conflict with widget key

    if 'debug_result' in st.session_state:
        result = st.session_state.debug_result
        code = st.session_state.stored_code  # ✅ use safe key

        with col2:
            st.markdown("**Execution Trace**")
            for log in result['execution_log']:
                st.code(f"Line {log['line']}", language='python')
                with st.expander("Variables at this point"):
                    st.json(log['variables'])

        st.markdown("### Execution Results")
        tab1, tab2, tab3 = st.tabs(["Output", "Variables", "Errors"])

        with tab1:
            st.code(result['output'] or "No output", language='text')

        with tab2:
            st.json(result['variables'])

        with tab3:
            if result['error']:
                st.error(result['error'])
            else:
                st.success("No errors detected")

        if result['execution_log']:
            st.markdown("### Step-through Debugger")
            if len(result['execution_log']) > 1:  # Only show slider if there are multiple steps
                current_step = st.slider(
                    "Execution step", 
                    0, 
                    len(result['execution_log']) - 1, 
                    0,
                    key="debug_step"
                )
            else:
                current_step = 0  # Default to first (and only) step
            
            selected_step = result['execution_log'][current_step]

            st.markdown(f"**Line {selected_step['line']}**")
            st.code(get_line_code(code, selected_step['line']), language='python')

            st.markdown("**Variable State**")
            st.json(selected_step['variables'])
        else:
            st.info("No execution steps recorded - the code may not have reached any instrumented lines")

# Run the sandbox UI
show_debug_sandbox()
