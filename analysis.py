from sos.sos import Learning, Analysis
from sos.validation import polynomial_check, visualise, lyapunov_check
from config.systems import NonlinearController, NonlinearSystem

xlims = [-10, 10]
ylims = [-10, 10]

if __name__ == "__main__":

    for system in {NonlinearController, NonlinearSystem}:

        analysis = Analysis(system.states, system.system, oV=system.oV,
                            verbose=False)
        lyapunov = analysis.lyapunov(printout=True)
        visualise(system.system, lyapunov, system.states, xlims=xlims,
                  ylims=ylims)

        # Set up learning and check that it's feasable
        sos = Learning(system.states, system.Z, system.A, system.B,
                       oP=system.oP, oF=system.oF, epsilon=0.1)
        is_feasable = sos.feasability_check(verbose=False)
        polynomial_check(sos, system.states, threshold=1E-5)
        system, lyapunov, _ = lyapunov_check(sos)
