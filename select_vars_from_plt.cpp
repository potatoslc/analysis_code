#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_PlotFileData.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>
#include <AMReX_String.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        ParmParse pp;

        std::string infile, outfile;
        pp.get("infile", infile);
        pp.get("outfile", outfile);

        // vars can be passed as:
        // vars="temperature pressure Y(H2)"
        Vector<std::string> vars;
        int nvars = pp.countval("vars");
        if (nvars <= 0) {
            amrex::Abort("No variables specified. Use vars=\"var1 var2 ...\"");
        }
        pp.getarr("vars", vars);

        // Read input plotfile
        PlotFileData pfd(infile);
        const int finest_level = pfd.finestLevel();

        const Vector<std::string>& all_names = pfd.varNames();
        const int n_all = static_cast<int>(all_names.size());

        // Map variable name -> component index
        std::unordered_map<std::string,int> name_to_comp;
        name_to_comp.reserve(all_names.size());
        for (int i = 0; i < n_all; ++i) {
            name_to_comp[all_names[i]] = i;
        }

        // Validate requested vars and build component list
        Vector<int> comps;
        comps.reserve(vars.size());
        for (const auto& v : vars) {
            auto it = name_to_comp.find(v);
            if (it == name_to_comp.end()) {
                std::string msg = "Requested variable not found in plotfile: " + v;
                amrex::Abort(msg);
            }
            comps.push_back(it->second);
        }

        // Ensure unique components (in case user repeated a variable)
        std::sort(comps.begin(), comps.end());
        comps.erase(std::unique(comps.begin(), comps.end()), comps.end());

        // Rebuild output var names in the same order as comps
        Vector<std::string> out_names;
        out_names.reserve(comps.size());
        for (int c : comps) {
            out_names.push_back(all_names[c]);
        }

        // Geometry / hierarchy metadata
        Vector<Geometry> geom(finest_level + 1);
        Vector<BoxArray> ba(finest_level + 1);
        Vector<DistributionMapping> dm(finest_level + 1);
        Vector<const iMultiFab*> masks(finest_level + 1, nullptr);

        Vector<int> level_steps(finest_level + 1, 0);
        Vector<IntVect> ref_ratio(finest_level + 1);

        for (int lev = 0; lev <= finest_level; ++lev) {
            geom[lev] = pfd.geom(lev);
            ba[lev]   = pfd.boxArray(lev);
            dm[lev]   = pfd.DistributionMap(lev);
            level_steps[lev] = pfd.levelStep(lev);
            if (lev < finest_level) {
                ref_ratio[lev] = pfd.refRatio(lev);
            }
        }

        // Allocate output data for each level
        Vector<std::unique_ptr<MultiFab>> out_mf(finest_level + 1);

        for (int lev = 0; lev <= finest_level; ++lev) {
            out_mf[lev] = std::make_unique<MultiFab>(ba[lev], dm[lev],
                                                     static_cast<int>(comps.size()), 0);

            MultiFab& dst = *out_mf[lev];

            // Copy each selected component from input plotfile to output MultiFab
            for (int n = 0; n < static_cast<int>(comps.size()); ++n) {
                const int src_comp = comps[n];
                pfd.fillVar(dst, lev, all_names[src_comp], n);
            }
        }

        // Build pointer vector expected by WriteMultiLevelPlotfile
        Vector<const MultiFab*> mf_ptr(finest_level + 1, nullptr);
        for (int lev = 0; lev <= finest_level; ++lev) {
            mf_ptr[lev] = out_mf[lev].get();
        }

        // Write reduced plotfile
        const auto& prob_domain = pfd.probDomain();
        const auto& prob_lo     = pfd.probLo();
        const auto& prob_hi     = pfd.probHi();
        const Real time         = pfd.time();

        amrex::WriteMultiLevelPlotfile(
            outfile,
            finest_level + 1,
            mf_ptr,
            out_names,
            geom,
            time,
            level_steps,
            ref_ratio
        );

        if (ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "Wrote reduced plotfile: " << outfile << "\n";
            amrex::Print() << "Kept variables:\n";
            for (const auto& n : out_names) {
                amrex::Print() << "  " << n << "\n";
            }
        }
    }
    amrex::Finalize();
    return 0;
}
