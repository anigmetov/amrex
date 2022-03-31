#include <AMReX_VisMF.H>
#include <AMReX_AsyncOut.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_FPC.H>
#include <AMReX_FabArrayUtility.H>

#ifdef AMREX_USE_EB
#include <AMReX_EBFabFactory.H>
#endif

#include "hdf5.h"

#ifdef AMREX_USE_HDF5_ZFP
#include "H5Zzfp_lib.h"
#include "H5Zzfp_props.h"
#endif

#ifdef AMREX_USE_HDF5_SZ
#include "H5Z_SZ.h"
#endif

#include <fstream>
#include <iomanip>

namespace amrex {

#ifdef AMREX_USE_HDF5_ASYNC
hid_t es_id_g = 0;
#endif

static int CreateWriteHDF5AttrDouble(hid_t loc, const char *name, hsize_t n, const double *data)
{
    herr_t ret;
    hid_t attr, attr_space;
    hsize_t dims = n;

    attr_space = H5Screate_simple(1, &dims, NULL);

    attr = H5Acreate(loc, name, H5T_NATIVE_DOUBLE, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) {
        printf("%s: Error with H5Acreate [%s]\n", __func__, name);
        return -1;
    }

    ret  = H5Awrite(attr, H5T_NATIVE_DOUBLE, (void*)data);
    if (ret < 0) {
        printf("%s: Error with H5Awrite [%s]\n", __func__, name);
        return -1;
    }
    H5Sclose(attr_space);
    H5Aclose(attr);
    return 1;
}

static int CreateWriteHDF5AttrInt(hid_t loc, const char *name, hsize_t n, const int *data)
{
    herr_t ret;
    hid_t attr, attr_space;
    hsize_t dims = n;

    attr_space = H5Screate_simple(1, &dims, NULL);

    attr = H5Acreate(loc, name, H5T_NATIVE_INT, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) {
        printf("%s: Error with H5Acreate [%s]\n", __func__, name);
        return -1;
    }

    ret  = H5Awrite(attr, H5T_NATIVE_INT, (void*)data);
    if (ret < 0) {
        printf("%s: Error with H5Awrite [%s]\n", __func__, name);
        return -1;
    }
    H5Sclose(attr_space);
    H5Aclose(attr);
    return 1;
}

static int CreateWriteHDF5AttrString(hid_t loc, const char *name, const char* str)
{
    hid_t attr, atype, space;
    herr_t ret;

    BL_ASSERT(name);
    BL_ASSERT(str);

    space = H5Screate(H5S_SCALAR);
    atype = H5Tcopy(H5T_C_S1);
    H5Tset_size(atype, strlen(str)+1);
    H5Tset_strpad(atype,H5T_STR_NULLTERM);
    attr = H5Acreate(loc, name, atype, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) {
        printf("%s: Error with H5Acreate [%s]\n", __func__, name);
        return -1;
    }

    ret = H5Awrite(attr, atype, str);
    if (ret < 0) {
        printf("%s: Error with H5Awrite[%s]\n", __func__, name);
        return -1;
    }

    H5Tclose(atype);
    H5Sclose(space);
    H5Aclose(attr);

    return 1;
}

#ifdef AMREX_USE_HDF5_ASYNC
static int CreateWriteHDF5AttrIntAsync(hid_t loc, const char *name, hsize_t n, const int *data)
{
    herr_t ret;
    hid_t attr, attr_space;
    hsize_t dims = n;

    attr_space = H5Screate_simple(1, &dims, NULL);

    attr = H5Acreate_async(loc, name, H5T_NATIVE_INT, attr_space, H5P_DEFAULT, H5P_DEFAULT, es_id_g);
    if (attr < 0) {
        printf("%s: Error with H5Acreate_async [%s]\n", __func__, name);
        return -1;
    }

    ret  = H5Awrite_async(attr, H5T_NATIVE_INT, (void*)data, es_id_g);
    if (ret < 0) {
        printf("%s: Error with H5Awrite_async [%s]\n", __func__, name);
        return -1;
    }
    H5Sclose(attr_space);
    H5Aclose_async(attr, es_id_g);

    return 1;
}

static int CreateWriteHDF5AttrDoubleAsync(hid_t loc, const char *name, hsize_t n, const double *data)
{
    herr_t ret;
    hid_t attr, attr_space;
    hsize_t dims = n;

    attr_space = H5Screate_simple(1, &dims, NULL);

    attr = H5Acreate_async(loc, name, H5T_NATIVE_DOUBLE, attr_space, H5P_DEFAULT, H5P_DEFAULT, es_id_g);
    if (attr < 0) {
        printf("%s: Error with H5Acreate [%s]\n", __func__, name);
        return -1;
    }

    ret  = H5Awrite_async(attr, H5T_NATIVE_DOUBLE, (void*)data, es_id_g);
    if (ret < 0) {
        printf("%s: Error with H5Awrite [%s]\n", __func__, name);
        return -1;
    }
    H5Sclose(attr_space);
    H5Aclose_async(attr, es_id_g);
    return 1;
}
#endif

#ifdef BL_USE_MPI
static void SetHDF5fapl(hid_t fapl, MPI_Comm comm)
#else
static void SetHDF5fapl(hid_t fapl)
#endif
{
#ifdef BL_USE_MPI
    H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL);

    // Alignment and metadata block size
    int alignment = 16 * 1024 * 1024;
    int blocksize =  4 * 1024 * 1024;
    H5Pset_alignment(fapl, alignment, alignment);
    H5Pset_meta_block_size(fapl, blocksize);

    // Collective metadata ops
    H5Pset_coll_metadata_write(fapl, true);
    H5Pset_all_coll_metadata_ops(fapl, true);

    // Defer cache flush
    H5AC_cache_config_t cache_config;
    cache_config.version = H5AC__CURR_CACHE_CONFIG_VERSION;
    H5Pget_mdc_config(fapl, &cache_config);
    cache_config.set_initial_size = 1;
    cache_config.initial_size = 16 * 1024 * 1024;
    cache_config.evictions_enabled = 0;
    cache_config.incr_mode = H5C_incr__off;
    cache_config.flash_incr_mode = H5C_flash_incr__off;
    cache_config.decr_mode = H5C_decr__off;
    H5Pset_mdc_config (fapl, &cache_config);
#else
    H5Pset_fapl_sec2(fapl);
#endif

}

static void
WriteGenericPlotfileHeaderHDF5 (hid_t fid,
                               int nlevels,
                               const Vector<const MultiFab*>& mf,
                               const Vector<BoxArray> &bArray,
                               const Vector<std::string> &varnames,
                               const Vector<Geometry> &geom,
                               Real time,
                               const Vector<int> &level_steps,
                               const Vector<IntVect> &ref_ratio,
                               const std::string &versionName,
                               const std::string &levelPrefix,
                               const std::string &mfPrefix,
                               const Vector<std::string>& extra_dirs)
{
    BL_PROFILE("WriteGenericPlotfileHeaderHDF5()");

    BL_ASSERT(nlevels <= bArray.size());
    BL_ASSERT(nlevels <= geom.size());
    BL_ASSERT(nlevels <= ref_ratio.size()+1);
    BL_ASSERT(nlevels <= level_steps.size());

    int finest_level(nlevels - 1);

    CreateWriteHDF5AttrString(fid, "version_name", versionName.c_str());
    CreateWriteHDF5AttrString(fid, "plotfile_type", "VanillaHDF5");

    int ncomp = varnames.size();
    CreateWriteHDF5AttrInt(fid, "num_components", 1, &ncomp);

    char comp_name[32];
    for (int ivar = 0; ivar < varnames.size(); ++ivar) {
        sprintf(comp_name, "component_%d", ivar);
        CreateWriteHDF5AttrString(fid, comp_name, varnames[ivar].c_str());
    }

    int ndim = AMREX_SPACEDIM;
    CreateWriteHDF5AttrInt(fid, "dim", 1, &ndim);
    double cur_time = (double)time;
    CreateWriteHDF5AttrDouble(fid, "time", 1, &cur_time);
    CreateWriteHDF5AttrInt(fid, "finest_level", 1, &finest_level);


    int coord = (int) geom[0].Coord();
    CreateWriteHDF5AttrInt(fid, "coordinate_system", 1, &coord);

    hid_t grp;
    char level_name[128];
    double lo[AMREX_SPACEDIM], hi[AMREX_SPACEDIM], cellsizes[AMREX_SPACEDIM];

    // For VisIt Chombo plot
    CreateWriteHDF5AttrInt(fid, "num_levels", 1, &nlevels);
    grp = H5Gcreate(fid, "Chombo_global", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CreateWriteHDF5AttrInt(grp, "SpaceDim", 1, &ndim);
    H5Gclose(grp);

    hid_t comp_dtype;

    comp_dtype = H5Tcreate (H5T_COMPOUND, 2 * AMREX_SPACEDIM * sizeof(int));
    if (1 == AMREX_SPACEDIM) {
        H5Tinsert (comp_dtype, "lo_i", 0 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (comp_dtype, "hi_i", 1 * sizeof(int), H5T_NATIVE_INT);
    }
    else if (2 == AMREX_SPACEDIM) {
        H5Tinsert (comp_dtype, "lo_i", 0 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (comp_dtype, "lo_j", 1 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (comp_dtype, "hi_i", 2 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (comp_dtype, "hi_j", 3 * sizeof(int), H5T_NATIVE_INT);
    }
    else if (3 == AMREX_SPACEDIM) {
        H5Tinsert (comp_dtype, "lo_i", 0 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (comp_dtype, "lo_j", 1 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (comp_dtype, "lo_k", 2 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (comp_dtype, "hi_i", 3 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (comp_dtype, "hi_j", 4 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (comp_dtype, "hi_k", 5 * sizeof(int), H5T_NATIVE_INT);
    }

    for (int level = 0; level <= finest_level; ++level) {
        sprintf(level_name, "level_%d", level);
        /* sprintf(level_name, "%s%d", levelPrefix.c_str(), level); */
        grp = H5Gcreate(fid, level_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (grp < 0) {
            std::cout << "H5Gcreate [" << level_name << "] failed!" << std::endl;
            continue;
        }

        int ratio = 1;
        if (ref_ratio.size() > 0)
            ratio = ref_ratio[level][0];

        if (level == finest_level) {
            ratio = 1;
        }
        CreateWriteHDF5AttrInt(grp, "ref_ratio", 1, &ratio);

        for (int k = 0; k < AMREX_SPACEDIM; ++k) {
            cellsizes[k] = (double)geom[level].CellSize()[k];
        }
        // Visit has issues with vec_dx, and is ok with a single "dx" value
        CreateWriteHDF5AttrDouble(grp, "Vec_dx", AMREX_SPACEDIM, cellsizes);
        // For VisIt Chombo plot
        CreateWriteHDF5AttrDouble(grp, "dx", 1, &cellsizes[0]);

        for (int i = 0; i < AMREX_SPACEDIM; ++i) {
            lo[i] = (double)geom[level].ProbLo(i);
            hi[i] = (double)geom[level].ProbHi(i);
        }
        CreateWriteHDF5AttrDouble(grp, "prob_lo", AMREX_SPACEDIM, lo);
        CreateWriteHDF5AttrDouble(grp, "prob_hi", AMREX_SPACEDIM, hi);

        int domain[AMREX_SPACEDIM*2];
        Box tmp(geom[level].Domain());
        for (int i = 0; i < AMREX_SPACEDIM; ++i) {
            domain[i] = tmp.smallEnd(i);
            domain[i+AMREX_SPACEDIM] = tmp.bigEnd(i);
        }

        hid_t aid = H5Screate(H5S_SCALAR);
        hid_t domain_attr = H5Acreate(grp, "prob_domain", comp_dtype, aid, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(domain_attr, comp_dtype, domain);
        H5Aclose(domain_attr);
        H5Sclose(aid);

        int type[AMREX_SPACEDIM];
        for (int i = 0; i < AMREX_SPACEDIM; ++i) {
            type[i] = (int)geom[level].Domain().ixType().test(i) ? 1 : 0;
        }
        CreateWriteHDF5AttrInt(grp, "domain_type", AMREX_SPACEDIM, type);

        CreateWriteHDF5AttrInt(grp, "steps", 1, &level_steps[level]);

        int ngrid = bArray[level].size();
        CreateWriteHDF5AttrInt(grp, "ngrid", 1, &ngrid);
        cur_time = (double)time;
        CreateWriteHDF5AttrDouble(grp, "time", 1, &cur_time);

        int ngrow = mf[level]->nGrow();
        CreateWriteHDF5AttrInt(grp, "ngrow", 1, &ngrow);

        /* hsize_t npts = ngrid*AMREX_SPACEDIM*2; */
        /* double *realboxes = new double [npts]; */
        /* for (int i = 0; i < bArray[level].size(); ++i) */
        /* { */
        /*     const Box &b(bArray[level][i]); */
        /*     RealBox loc = RealBox(b, geom[level].CellSize(), geom[level].ProbLo()); */
        /*     for (int n = 0; n < AMREX_SPACEDIM; ++n) { */
        /*         /1* HeaderFile << loc.lo(n) << ' ' << loc.hi(n) << '\n'; *1/ */
        /*         realboxes[i*AMREX_SPACEDIM*2 + n] = loc.lo(n); */
        /*         realboxes[i*AMREX_SPACEDIM*2 + AMREX_SPACEDIM + n] = loc.hi(n); */
        /*     } */
        /* } */
        /* CreateWriteDsetDouble(grp, "Boxes", npts, realboxes); */
        /* delete [] realboxes; */

        H5Gclose(grp);
    }

    H5Tclose(comp_dtype);
}

#ifdef AMREX_USE_HDF5_ASYNC
void async_vol_es_wait_close()
{
    size_t num_in_progress;
    hbool_t op_failed;
    if (es_id_g != 0) {
        H5ESwait(es_id_g, H5ES_WAIT_FOREVER, &num_in_progress, &op_failed);
        if (num_in_progress != 0)
            std::cout << "After H5ESwait, still has async operations in progress!" << std::endl;
        H5ESclose(es_id_g);
        es_id_g = 0;
        /* std::cout << "es_id_g closed!" << std::endl; */
    }
    return;
}
static void async_vol_es_wait()
{
    size_t num_in_progress;
    hbool_t op_failed;
    if (es_id_g != 0) {
        H5ESwait(es_id_g, H5ES_WAIT_FOREVER, &num_in_progress, &op_failed);
        if (num_in_progress != 0)
            std::cout << "After H5ESwait, still has async operations in progress!" << std::endl;
    }
    return;
}
#endif

void WriteMultiLevelPlotfileHDF5SingleDset (const std::string& plotfilename,
                                            int nlevels,
                                            const Vector<const MultiFab*>& mf,
                                            const Vector<std::string>& varnames,
                                            const Vector<Geometry>& geom,
                                            Real time,
                                            const Vector<int>& level_steps,
                                            const Vector<IntVect>& ref_ratio,
                                            const std::string &compression,
                                            const std::string &versionName,
                                            const std::string &levelPrefix,
                                            const std::string &mfPrefix,
                                            const Vector<std::string>& extra_dirs)
{
    BL_PROFILE("WriteMultiLevelPlotfileHDF5SingleDset");

    BL_ASSERT(nlevels <= mf.size());
    BL_ASSERT(nlevels <= geom.size());
    BL_ASSERT(nlevels <= ref_ratio.size()+1);
    BL_ASSERT(nlevels <= level_steps.size());
    BL_ASSERT(mf[0]->nComp() == varnames.size());

    int myProc(ParallelDescriptor::MyProc());
    int nProcs(ParallelDescriptor::NProcs());

#ifdef AMREX_USE_HDF5_ASYNC
    // For HDF5 async VOL, block and wait previous tasks have all completed
    if (es_id_g != 0) {
        async_vol_es_wait();
    }
    else {
        ExecOnFinalize(async_vol_es_wait_close);
        es_id_g = H5EScreate();
    }
#endif

    herr_t  ret;
    int finest_level = nlevels-1;
    int ncomp = mf[0]->nComp();
    /* double total_write_start_time(ParallelDescriptor::second()); */
    std::string filename(plotfilename + ".h5");

    // Write out root level metadata
    hid_t fapl, dxpl_col, dxpl_ind, dcpl_id, fid, grp;

    if(ParallelDescriptor::IOProcessor()) {
        BL_PROFILE_VAR("H5writeMetadata", h5dwm);
        // Create the HDF5 file
        fid = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (fid < 0)
            FileOpenFailed(filename.c_str());

        Vector<BoxArray> boxArrays(nlevels);
        for(int level(0); level < boxArrays.size(); ++level) {
            boxArrays[level] = mf[level]->boxArray();
        }

        WriteGenericPlotfileHeaderHDF5(fid, nlevels, mf, boxArrays, varnames, geom, time, level_steps, ref_ratio, versionName, levelPrefix, mfPrefix, extra_dirs);
        H5Fclose(fid);
        BL_PROFILE_VAR_STOP(h5dwm);
    }

    ParallelDescriptor::Barrier();

    hid_t babox_id;
    babox_id = H5Tcreate (H5T_COMPOUND, 2 * AMREX_SPACEDIM * sizeof(int));
    if (1 == AMREX_SPACEDIM) {
        H5Tinsert (babox_id, "lo_i", 0 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (babox_id, "hi_i", 1 * sizeof(int), H5T_NATIVE_INT);
    }
    else if (2 == AMREX_SPACEDIM) {
        H5Tinsert (babox_id, "lo_i", 0 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (babox_id, "lo_j", 1 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (babox_id, "hi_i", 2 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (babox_id, "hi_j", 3 * sizeof(int), H5T_NATIVE_INT);
    }
    else if (3 == AMREX_SPACEDIM) {
        H5Tinsert (babox_id, "lo_i", 0 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (babox_id, "lo_j", 1 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (babox_id, "lo_k", 2 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (babox_id, "hi_i", 3 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (babox_id, "hi_j", 4 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (babox_id, "hi_k", 5 * sizeof(int), H5T_NATIVE_INT);
    }

    hid_t center_id = H5Tcreate (H5T_COMPOUND, AMREX_SPACEDIM * sizeof(int));
    if (1 == AMREX_SPACEDIM) {
        H5Tinsert (center_id, "i", 0 * sizeof(int), H5T_NATIVE_INT);
    }
    else if (2 == AMREX_SPACEDIM) {
        H5Tinsert (center_id, "i", 0 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (center_id, "j", 1 * sizeof(int), H5T_NATIVE_INT);
    }
    else if (3 == AMREX_SPACEDIM) {
        H5Tinsert (center_id, "i", 0 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (center_id, "j", 1 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (center_id, "k", 2 * sizeof(int), H5T_NATIVE_INT);
    }

    fapl = H5Pcreate (H5P_FILE_ACCESS);
    dxpl_col = H5Pcreate(H5P_DATASET_XFER);
    dxpl_ind = H5Pcreate(H5P_DATASET_XFER);

#ifdef BL_USE_MPI
    SetHDF5fapl(fapl, ParallelDescriptor::Communicator());
    H5Pset_dxpl_mpio(dxpl_col, H5FD_MPIO_COLLECTIVE);
#else
    SetHDF5fapl(fapl);
#endif

    dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_fill_time(dcpl_id, H5D_FILL_TIME_NEVER);

#if (defined AMREX_USE_HDF5_ZFP) || (defined AMREX_USE_HDF5_SZ)
    const char *chunk_env = NULL;
    std::string mode_env, value_env;
    double comp_value = -1.0;
    hsize_t chunk_dim = 1024;

    chunk_env = getenv("HDF5_CHUNK_SIZE");
    if (chunk_env != NULL)
        chunk_dim = atoi(chunk_env);

    H5Pset_chunk(dcpl_id, 1, &chunk_dim);
    H5Pset_alloc_time(dcpl_id, H5D_ALLOC_TIME_LATE);

    std::string::size_type pos = compression.find('@');
    if (pos != std::string::npos) {
        mode_env = compression.substr(0, pos);
        value_env = compression.substr(pos+1);
        if (!value_env.empty()) {
            comp_value = atof(value_env.c_str());
        }
    }

#ifdef AMREX_USE_HDF5_ZFP
    pos = compression.find("ZFP");
    if (pos != std::string::npos) {
        ret = H5Z_zfp_initialize();
        if (ret < 0) amrex::Abort("ZFP initialize failed!");
    }
#endif

#ifdef AMREX_USE_HDF5_SZ
    pos = compression.find("SZ");
    if (pos != std::string::npos) {
        ret = H5Z_SZ_Init((char*)value_env.c_str());
        if (ret < 0) {
            std::cout << "SZ config file:" << value_env.c_str() << std::endl;
            amrex::Abort("SZ initialize failed, check SZ config file!");
        }
    }
#endif

    if (!mode_env.empty() && mode_env != "None") {
        if (mode_env == "ZLIB")
            H5Pset_deflate(dcpl_id, (int)comp_value);
#ifdef AMREX_USE_HDF5_ZFP
        else if (mode_env == "ZFP_RATE")
            H5Pset_zfp_rate(dcpl_id, comp_value);
        else if (mode_env == "ZFP_PRECISION")
            H5Pset_zfp_precision(dcpl_id, (unsigned int)comp_value);
        else if (mode_env == "ZFP_ACCURACY")
            H5Pset_zfp_accuracy(dcpl_id, comp_value);
        else if (mode_env == "ZFP_REVERSIBLE")
            H5Pset_zfp_reversible(dcpl_id);
        else if (mode_env == "ZLIB")
            H5Pset_deflate(dcpl_id, (int)comp_value);
#endif

        if (ParallelDescriptor::MyProc() == 0) {
            std::cout << "\nHDF5 plotfile using " << mode_env << ", " <<
                value_env << ", " << chunk_dim << std::endl;
        }
    }
#endif

    BL_PROFILE_VAR("H5writeAllLevel", h5dwd);

    // All process open the file
#ifdef AMREX_USE_HDF5_ASYNC
    // Only use async for writing actual data
    fid = H5Fopen_async(filename.c_str(), H5F_ACC_RDWR, fapl, es_id_g);
#else
    fid = H5Fopen(filename.c_str(), H5F_ACC_RDWR, fapl);
#endif
    if (fid < 0)
        FileOpenFailed(filename.c_str());

    auto whichRD = FArrayBox::getDataDescriptor();
    bool doConvert(*whichRD != FPC::NativeRealDescriptor());
    int whichRDBytes(whichRD->numBytes());

    // Write data for each level
    char level_name[32];
    for (int level = 0; level <= finest_level; ++level) {
        sprintf(level_name, "level_%d", level);
#ifdef AMREX_USE_HDF5_ASYNC
        grp = H5Gopen_async(fid, level_name, H5P_DEFAULT, es_id_g);
#else
        grp = H5Gopen(fid, level_name, H5P_DEFAULT);
#endif
        if (grp < 0) { std::cout << "H5Gopen [" << level_name << "] failed!" << std::endl; break; }

        // Get the boxes assigned to all ranks and calculate their offsets and sizes
        Vector<int> procMap = mf[level]->DistributionMap().ProcessorMap();
        const BoxArray& grids = mf[level]->boxArray();
        hid_t boxdataset, boxdataspace;
        hid_t offsetdataset, offsetdataspace;
        hid_t centerdataset, centerdataspace;
        std::string bdsname("boxes");
        std::string odsname("data:offsets=0");
        std::string centername("boxcenter");
        std::string dataname("data:datatype=0");
        hsize_t  flatdims[1];
        flatdims[0] = grids.size();

        flatdims[0] = grids.size();
        boxdataspace = H5Screate_simple(1, flatdims, NULL);

#ifdef AMREX_USE_HDF5_ASYNC
        boxdataset = H5Dcreate_async(grp, bdsname.c_str(), babox_id, boxdataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT, es_id_g);
#else
        boxdataset = H5Dcreate(grp, bdsname.c_str(), babox_id, boxdataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#endif
        if (boxdataset < 0) { std::cout << "H5Dcreate [" << bdsname << "] failed!" << std::endl; break; }

        // Create a boxarray sorted by rank
        std::map<int, Vector<Box> > gridMap;
        for(int i(0); i < grids.size(); ++i) {
            int gridProc(procMap[i]);
            Vector<Box> &boxesAtProc = gridMap[gridProc];
            boxesAtProc.push_back(grids[i]);
        }
        BoxArray sortedGrids(grids.size());
        Vector<int> sortedProcs(grids.size());
        int bIndex(0);
        for(auto it = gridMap.begin(); it != gridMap.end(); ++it) {
            int proc = it->first;
            Vector<Box> &boxesAtProc = it->second;
            for(int ii(0); ii < boxesAtProc.size(); ++ii) {
                sortedGrids.set(bIndex, boxesAtProc[ii]);
                sortedProcs[bIndex] = proc;
                ++bIndex;
            }
        }

        hsize_t  oflatdims[1];
        oflatdims[0] = sortedGrids.size() + 1;
        offsetdataspace = H5Screate_simple(1, oflatdims, NULL);
#ifdef AMREX_USE_HDF5_ASYNC
        offsetdataset = H5Dcreate_async(grp, odsname.c_str(), H5T_NATIVE_LLONG, offsetdataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT, es_id_g);
#else
        offsetdataset = H5Dcreate(grp, odsname.c_str(), H5T_NATIVE_LLONG, offsetdataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#endif
        if(offsetdataset < 0) { std::cout << "create offset dataset failed! ret = " << offsetdataset << std::endl; break;}

        hsize_t centerdims[1];
        centerdims[0]   = sortedGrids.size() ;
        centerdataspace = H5Screate_simple(1, centerdims, NULL);
#ifdef AMREX_USE_HDF5_ASYNC
        centerdataset = H5Dcreate_async(grp, centername.c_str(), center_id, centerdataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT, es_id_g);
#else
        centerdataset = H5Dcreate(grp, centername.c_str(), center_id, centerdataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#endif
        if(centerdataset < 0) { std::cout << "Create center dataset failed! ret = " << centerdataset << std::endl; break;}

        Vector<unsigned long long> offsets(sortedGrids.size() + 1, 0);
        unsigned long long currentOffset(0L);
        for(int b(0); b < sortedGrids.size(); ++b) {
            offsets[b] = currentOffset;
            currentOffset += sortedGrids[b].numPts() * ncomp;
        }
        offsets[sortedGrids.size()] = currentOffset;

        Vector<unsigned long long> procOffsets(nProcs, 0);
        Vector<unsigned long long> procBufferSize(nProcs, 0);
        unsigned long long totalOffset(0);
        for(auto it = gridMap.begin(); it != gridMap.end(); ++it) {
            int proc = it->first;
            Vector<Box> &boxesAtProc = it->second;
            procOffsets[proc] = totalOffset;
            procBufferSize[proc] = 0L;
            for(int b(0); b < boxesAtProc.size(); ++b) {
                procBufferSize[proc] += boxesAtProc[b].numPts() * ncomp;
            }
            totalOffset += procBufferSize[proc];

            /* if (level == 2) { */
            /*     fprintf(stderr, "Rank %d: level %d, proc %d, offset %ld, size %ld, all size %ld\n", */
            /*             myProc, level, proc, procOffsets[proc], procBufferSize[proc], totalOffset); */
            /* } */
        }

        if(ParallelDescriptor::IOProcessor()) {
            int vbCount(0);
            Vector<int> vbox(sortedGrids.size() * 2 * AMREX_SPACEDIM);
            Vector<int> centering(sortedGrids.size() * AMREX_SPACEDIM);
            for(int b(0); b < sortedGrids.size(); ++b) {
                for(int i(0); i < AMREX_SPACEDIM; ++i) {
                    vbox[(vbCount * 2 * AMREX_SPACEDIM) + i] = sortedGrids[b].smallEnd(i);
                    vbox[(vbCount * 2 * AMREX_SPACEDIM) + i + AMREX_SPACEDIM] = sortedGrids[b].bigEnd(i);
                    centering[vbCount * AMREX_SPACEDIM + i] = sortedGrids[b].ixType().test(i) ? 1 : 0;
                }
                ++vbCount;
            }

            // Only proc zero needs to write out this information
#ifdef AMREX_USE_HDF5_ASYNC
            ret = H5Dwrite_async(offsetdataset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, dxpl_ind, &(offsets[0]), es_id_g);
#else
            ret = H5Dwrite(offsetdataset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, dxpl_ind, &(offsets[0]));
#endif
            if(ret < 0) { std::cout << "Write offset dataset failed! ret = " << ret << std::endl; }

#ifdef AMREX_USE_HDF5_ASYNC
            ret = H5Dwrite_async(centerdataset, center_id, H5S_ALL, H5S_ALL, dxpl_ind, &(centering[0]), es_id_g);
#else
            ret = H5Dwrite(centerdataset, center_id, H5S_ALL, H5S_ALL, dxpl_ind, &(centering[0]));
#endif
            if(ret < 0) { std::cout << "Write center dataset failed! ret = " << ret << std::endl; }

#ifdef AMREX_USE_HDF5_ASYNC
            ret = H5Dwrite_async(boxdataset, babox_id, H5S_ALL, H5S_ALL, dxpl_ind, &(vbox[0]), es_id_g);
#else
            ret = H5Dwrite(boxdataset, babox_id, H5S_ALL, H5S_ALL, dxpl_ind, &(vbox[0]));
#endif
            if(ret < 0) { std::cout << "Write box dataset failed! ret = " << ret << std::endl; }
        } // end IOProcessor

        hsize_t hs_procsize[1], hs_allprocsize[1], ch_offset[1];

        ch_offset[0]       = procOffsets[myProc];          // ---- offset on this proc
        hs_procsize[0]     = procBufferSize[myProc];       // ---- size of buffer on this proc
        hs_allprocsize[0]  = offsets[sortedGrids.size()];  // ---- size of buffer on all procs

        hid_t dataspace    = H5Screate_simple(1, hs_allprocsize, NULL);
        hid_t memdataspace = H5Screate_simple(1, hs_procsize, NULL);

        /* fprintf(stderr, "Rank %d: level %d, offset %ld, size %ld, all size %ld\n", myProc, level, ch_offset[0], hs_procsize[0], hs_allprocsize[0]); */

        if (hs_procsize[0] == 0)
            H5Sselect_none(dataspace);
        else
            H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, ch_offset, NULL, hs_procsize, NULL);

        Vector<Real> a_buffer(procBufferSize[myProc], -1.0);
        const MultiFab* data;
        std::unique_ptr<MultiFab> mf_tmp;
        if (mf[level]->nGrowVect() != 0) {
            mf_tmp = std::make_unique<MultiFab>(mf[level]->boxArray(),
                                                mf[level]->DistributionMap(),
                                                mf[level]->nComp(), 0, MFInfo(),
                                                mf[level]->Factory());
            MultiFab::Copy(*mf_tmp, *mf[level], 0, 0, mf[level]->nComp(), 0);
            data = mf_tmp.get();
        } else {
            data = mf[level];
        }

        Long writeDataItems(0), writeDataSize(0);
        for(MFIter mfi(*data); mfi.isValid(); ++mfi) {
            const FArrayBox &fab = (*data)[mfi];
            writeDataItems = fab.box().numPts() * (*data).nComp();
            if(doConvert) {
                RealDescriptor::convertFromNativeFormat(static_cast<void *> (a_buffer.dataPtr()+writeDataSize),
                                                        writeDataItems, fab.dataPtr(), *whichRD);
            } else {    // ---- copy from the fab
                memcpy(static_cast<void *> (a_buffer.dataPtr()+writeDataSize),
                       fab.dataPtr(), writeDataItems * whichRDBytes);
            }
            writeDataSize += writeDataItems;
        }

        BL_PROFILE_VAR("H5DwriteData", h5dwg);

#ifdef AMREX_USE_HDF5_SZ
        if (mode_env == "SZ") {
            size_t cd_nelmts;
            unsigned int* cd_values = NULL;
            unsigned filter_config;
            SZ_metaDataToCdArray(&cd_nelmts, &cd_values, SZ_DOUBLE, 0, 0, 0, 0, hs_allprocsize[0]);
            H5Pset_filter(dcpl_id, H5Z_FILTER_SZ, H5Z_FLAG_MANDATORY, cd_nelmts, cd_values);
        }
#endif

#ifdef AMREX_USE_HDF5_ASYNC
        hid_t dataset = H5Dcreate_async(grp, dataname.c_str(), H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, dcpl_id, H5P_DEFAULT, es_id_g);
#else
        hid_t dataset = H5Dcreate(grp, dataname.c_str(), H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
#endif
        if(dataset < 0)
            std::cout << ParallelDescriptor::MyProc() << "create data failed!  ret = " << dataset << std::endl;

#ifdef AMREX_USE_HDF5_ASYNC
        ret = H5Dwrite_async(dataset, H5T_NATIVE_DOUBLE, memdataspace, dataspace, dxpl_col, a_buffer.dataPtr(), es_id_g);
#else
        ret = H5Dwrite(dataset, H5T_NATIVE_DOUBLE, memdataspace, dataspace, dxpl_col, a_buffer.dataPtr());
#endif
        if(ret < 0) { std::cout << ParallelDescriptor::MyProc() << "Write data failed!  ret = " << ret << std::endl; break; }

        BL_PROFILE_VAR_STOP(h5dwg);

        H5Sclose(memdataspace);
        H5Sclose(dataspace);
        H5Sclose(offsetdataspace);
        H5Sclose(centerdataspace);
        H5Sclose(boxdataspace);

#ifdef AMREX_USE_HDF5_ASYNC
        H5Dclose_async(dataset, es_id_g);
        H5Dclose_async(offsetdataset, es_id_g);
        H5Dclose_async(centerdataset, es_id_g);
        H5Dclose_async(boxdataset, es_id_g);
        H5Gclose_async(grp, es_id_g);
#else
        H5Dclose(dataset);
        H5Dclose(offsetdataset);
        H5Dclose(centerdataset);
        H5Dclose(boxdataset);
        H5Gclose(grp);
#endif
    } // For group

    BL_PROFILE_VAR_STOP(h5dwd);

    H5Tclose(center_id);
    H5Tclose(babox_id);
    H5Pclose(fapl);
    H5Pclose(dxpl_col);
    H5Pclose(dxpl_ind);
    H5Pclose(dcpl_id);

#ifdef AMREX_USE_HDF5_ASYNC
    H5Fclose_async(fid, es_id_g);
#else
    H5Fclose(fid);
#endif
} // WriteMultiLevelPlotfileHDF5SingleDset

void WriteMultiLevelPlotfileHDF5MultiDset (const std::string& plotfilename,
                                           int nlevels,
                                           const Vector<const MultiFab*>& mf,
                                           const Vector<std::string>& varnames,
                                           const Vector<Geometry>& geom,
                                           Real time,
                                           const Vector<int>& level_steps,
                                           const Vector<IntVect>& ref_ratio,
                                           const std::string &compression,
                                           const std::string &versionName,
                                           const std::string &levelPrefix,
                                           const std::string &mfPrefix,
                                           const Vector<std::string>& extra_dirs)
{
    BL_PROFILE("WriteMultiLevelPlotfileHDF5MultiDset");

    BL_ASSERT(nlevels <= mf.size());
    BL_ASSERT(nlevels <= geom.size());
    BL_ASSERT(nlevels <= ref_ratio.size()+1);
    BL_ASSERT(nlevels <= level_steps.size());
    BL_ASSERT(mf[0]->nComp() == varnames.size());

    int myProc(ParallelDescriptor::MyProc());
    int nProcs(ParallelDescriptor::NProcs());

#ifdef AMREX_USE_HDF5_ASYNC
    // For HDF5 async VOL, block and wait previous tasks have all completed
    if (es_id_g != 0) {
        async_vol_es_wait();
    }
    else {
        ExecOnFinalize(async_vol_es_wait_close);
        es_id_g = H5EScreate();
    }
#endif

    herr_t  ret;
    int finest_level = nlevels-1;
    int ncomp = mf[0]->nComp();
    /* double total_write_start_time(ParallelDescriptor::second()); */
    std::string filename(plotfilename + ".h5");

    // Write out root level metadata
    hid_t fapl, dxpl_col, dxpl_ind, fid, grp, dcpl_id;

    if(ParallelDescriptor::IOProcessor()) {
        BL_PROFILE_VAR("H5writeMetadata", h5dwm);
        // Create the HDF5 file
        fid = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (fid < 0)
            FileOpenFailed(filename.c_str());

        Vector<BoxArray> boxArrays(nlevels);
        for(int level(0); level < boxArrays.size(); ++level) {
            boxArrays[level] = mf[level]->boxArray();
        }

        WriteGenericPlotfileHeaderHDF5(fid, nlevels, mf, boxArrays, varnames, geom, time, level_steps, ref_ratio, versionName, levelPrefix, mfPrefix, extra_dirs);
        H5Fclose(fid);
        BL_PROFILE_VAR_STOP(h5dwm);
    }

    ParallelDescriptor::Barrier();

    hid_t babox_id;
    babox_id = H5Tcreate (H5T_COMPOUND, 2 * AMREX_SPACEDIM * sizeof(int));
    if (1 == AMREX_SPACEDIM) {
        H5Tinsert (babox_id, "lo_i", 0 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (babox_id, "hi_i", 1 * sizeof(int), H5T_NATIVE_INT);
    }
    else if (2 == AMREX_SPACEDIM) {
        H5Tinsert (babox_id, "lo_i", 0 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (babox_id, "lo_j", 1 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (babox_id, "hi_i", 2 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (babox_id, "hi_j", 3 * sizeof(int), H5T_NATIVE_INT);
    }
    else if (3 == AMREX_SPACEDIM) {
        H5Tinsert (babox_id, "lo_i", 0 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (babox_id, "lo_j", 1 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (babox_id, "lo_k", 2 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (babox_id, "hi_i", 3 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (babox_id, "hi_j", 4 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (babox_id, "hi_k", 5 * sizeof(int), H5T_NATIVE_INT);
    }

    hid_t center_id = H5Tcreate (H5T_COMPOUND, AMREX_SPACEDIM * sizeof(int));
    if (1 == AMREX_SPACEDIM) {
        H5Tinsert (center_id, "i", 0 * sizeof(int), H5T_NATIVE_INT);
    }
    else if (2 == AMREX_SPACEDIM) {
        H5Tinsert (center_id, "i", 0 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (center_id, "j", 1 * sizeof(int), H5T_NATIVE_INT);
    }
    else if (3 == AMREX_SPACEDIM) {
        H5Tinsert (center_id, "i", 0 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (center_id, "j", 1 * sizeof(int), H5T_NATIVE_INT);
        H5Tinsert (center_id, "k", 2 * sizeof(int), H5T_NATIVE_INT);
    }

    fapl = H5Pcreate (H5P_FILE_ACCESS);
    dxpl_col = H5Pcreate(H5P_DATASET_XFER);
    dxpl_ind = H5Pcreate(H5P_DATASET_XFER);

#ifdef BL_USE_MPI
    SetHDF5fapl(fapl, ParallelDescriptor::Communicator());
    H5Pset_dxpl_mpio(dxpl_col, H5FD_MPIO_COLLECTIVE);
#else
    SetHDF5fapl(fapl);
#endif

    dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_fill_time(dcpl_id, H5D_FILL_TIME_NEVER);

#if (defined AMREX_USE_HDF5_ZFP) || (defined AMREX_USE_HDF5_SZ)
    const char *chunk_env = NULL;
    std::string mode_env, value_env;
    double comp_value = -1.0;
    hsize_t chunk_dim = 1024;

    chunk_env = getenv("HDF5_CHUNK_SIZE");
    if (chunk_env != NULL)
        chunk_dim = atoi(chunk_env);

    H5Pset_chunk(dcpl_id, 1, &chunk_dim);
    H5Pset_alloc_time(dcpl_id, H5D_ALLOC_TIME_LATE);

    std::string::size_type pos = compression.find('@');
    if (pos != std::string::npos) {
        mode_env = compression.substr(0, pos);
        value_env = compression.substr(pos+1);
        if (!value_env.empty()) {
            comp_value = atof(value_env.c_str());
        }
    }

#ifdef AMREX_USE_HDF5_ZFP
    pos = compression.find("ZFP");
    if (pos != std::string::npos) {
        ret = H5Z_zfp_initialize();
        if (ret < 0) amrex::Abort("ZFP initialize failed!");
    }
#endif

#ifdef AMREX_USE_HDF5_SZ
    pos = compression.find("SZ");
    if (pos != std::string::npos) {
        ret = H5Z_SZ_Init((char*)value_env.c_str());
        if (ret < 0) amrex::Abort("ZFP initialize failed, check SZ config file!");
    }
#endif

    if (!mode_env.empty() && mode_env != "None") {
        if (mode_env == "ZLIB")
            H5Pset_deflate(dcpl_id, (int)comp_value);
#ifdef AMREX_USE_HDF5_ZFP
        else if (mode_env == "ZFP_RATE")
            H5Pset_zfp_rate(dcpl_id, comp_value);
        else if (mode_env == "ZFP_PRECISION")
            H5Pset_zfp_precision(dcpl_id, (unsigned int)comp_value);
        else if (mode_env == "ZFP_ACCURACY")
            H5Pset_zfp_accuracy(dcpl_id, comp_value);
        else if (mode_env == "ZFP_REVERSIBLE")
            H5Pset_zfp_reversible(dcpl_id);
#endif

        if (ParallelDescriptor::MyProc() == 0) {
            std::cout << "\nHDF5 checkpoint using " << mode_env << ", " <<
                value_env << ", " << chunk_dim << std::endl;
        }
    }
#endif

    BL_PROFILE_VAR("H5writeAllLevel", h5dwd);

    // All process open the file
#ifdef AMREX_USE_HDF5_ASYNC
    // Only use async for writing actual data
    fid = H5Fopen_async(filename.c_str(), H5F_ACC_RDWR, fapl, es_id_g);
#else
    fid = H5Fopen(filename.c_str(), H5F_ACC_RDWR, fapl);
#endif
    if (fid < 0)
        FileOpenFailed(filename.c_str());

    auto whichRD = FArrayBox::getDataDescriptor();
    bool doConvert(*whichRD != FPC::NativeRealDescriptor());
    int whichRDBytes(whichRD->numBytes());

    // Write data for each level
    char level_name[32];

    for (int level = 0; level <= finest_level; ++level) {
        sprintf(level_name, "level_%d", level);
#ifdef AMREX_USE_HDF5_ASYNC
        grp = H5Gopen_async(fid, level_name, H5P_DEFAULT, es_id_g);
#else
        grp = H5Gopen(fid, level_name, H5P_DEFAULT);
#endif
        if (grp < 0) { std::cout << "H5Gopen [" << level_name << "] failed!" << std::endl; break; }

        // Get the boxes assigned to all ranks and calculate their offsets and sizes
        Vector<int> procMap = mf[level]->DistributionMap().ProcessorMap();
        const BoxArray& grids = mf[level]->boxArray();
        hid_t boxdataset, boxdataspace;
        hid_t offsetdataset, offsetdataspace;
        hid_t centerdataset, centerdataspace;
        std::string bdsname("boxes");
        std::string odsname("data:offsets=0");
        std::string centername("boxcenter");
        hsize_t  flatdims[1];
        flatdims[0] = grids.size();

        flatdims[0] = grids.size();
        boxdataspace = H5Screate_simple(1, flatdims, NULL);

#ifdef AMREX_USE_HDF5_ASYNC
        boxdataset = H5Dcreate_async(grp, bdsname.c_str(), babox_id, boxdataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT, es_id_g);
#else
        boxdataset = H5Dcreate(grp, bdsname.c_str(), babox_id, boxdataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#endif
        if (boxdataset < 0) { std::cout << "H5Dcreate [" << bdsname << "] failed!" << std::endl; break; }

        // Create a boxarray sorted by rank
        std::map<int, Vector<Box> > gridMap;
        for(int i(0); i < grids.size(); ++i) {
            int gridProc(procMap[i]);
            Vector<Box> &boxesAtProc = gridMap[gridProc];
            boxesAtProc.push_back(grids[i]);
        }
        BoxArray sortedGrids(grids.size());
        Vector<int> sortedProcs(grids.size());
        int bIndex(0);
        for(auto it = gridMap.begin(); it != gridMap.end(); ++it) {
            int proc = it->first;
            Vector<Box> &boxesAtProc = it->second;
            for(int ii(0); ii < boxesAtProc.size(); ++ii) {
                sortedGrids.set(bIndex, boxesAtProc[ii]);
                sortedProcs[bIndex] = proc;
                ++bIndex;
            }
        }

        hsize_t  oflatdims[1];
        oflatdims[0] = sortedGrids.size() + 1;
        offsetdataspace = H5Screate_simple(1, oflatdims, NULL);
#ifdef AMREX_USE_HDF5_ASYNC
        offsetdataset = H5Dcreate_async(grp, odsname.c_str(), H5T_NATIVE_LLONG, offsetdataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT, es_id_g);
#else
        offsetdataset = H5Dcreate(grp, odsname.c_str(), H5T_NATIVE_LLONG, offsetdataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#endif
        if(offsetdataset < 0) { std::cout << "create offset dataset failed! ret = " << offsetdataset << std::endl; break;}

        hsize_t centerdims[1];
        centerdims[0]   = sortedGrids.size() ;
        centerdataspace = H5Screate_simple(1, centerdims, NULL);
#ifdef AMREX_USE_HDF5_ASYNC
        centerdataset = H5Dcreate_async(grp, centername.c_str(), center_id, centerdataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT, es_id_g);
#else
        centerdataset = H5Dcreate(grp, centername.c_str(), center_id, centerdataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#endif
        if(centerdataset < 0) { std::cout << "Create center dataset failed! ret = " << centerdataset << std::endl; break;}

        Vector<unsigned long long> offsets(sortedGrids.size() + 1);
        unsigned long long currentOffset(0L);
        for(int b(0); b < sortedGrids.size(); ++b) {
            offsets[b] = currentOffset;
            /* currentOffset += sortedGrids[b].numPts() * ncomp; */
            currentOffset += sortedGrids[b].numPts();
        }
        offsets[sortedGrids.size()] = currentOffset;

        Vector<unsigned long long> procOffsets(nProcs);
        Vector<unsigned long long> procBufferSize(nProcs);
        unsigned long long totalOffset(0);
        for(auto it = gridMap.begin(); it != gridMap.end(); ++it) {
            int proc = it->first;
            Vector<Box> &boxesAtProc = it->second;
            procOffsets[proc] = totalOffset;
            procBufferSize[proc] = 0L;
            for(int b(0); b < boxesAtProc.size(); ++b) {
                /* procBufferSize[proc] += boxesAtProc[b].numPts() * ncomp; */
                procBufferSize[proc] += boxesAtProc[b].numPts();
            }
            totalOffset += procBufferSize[proc];
        }

        if(ParallelDescriptor::IOProcessor()) {
            int vbCount(0);
            Vector<int> vbox(sortedGrids.size() * 2 * AMREX_SPACEDIM);
            Vector<int> centering(sortedGrids.size() * AMREX_SPACEDIM);
            for(int b(0); b < sortedGrids.size(); ++b) {
                for(int i(0); i < AMREX_SPACEDIM; ++i) {
                    vbox[(vbCount * 2 * AMREX_SPACEDIM) + i] = sortedGrids[b].smallEnd(i);
                    vbox[(vbCount * 2 * AMREX_SPACEDIM) + i + AMREX_SPACEDIM] = sortedGrids[b].bigEnd(i);
                    centering[vbCount * AMREX_SPACEDIM + i] = sortedGrids[b].ixType().test(i) ? 1 : 0;
                }
                ++vbCount;
            }

            // Only proc zero needs to write out this information
#ifdef AMREX_USE_HDF5_ASYNC
            ret = H5Dwrite_async(offsetdataset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, dxpl_ind, &(offsets[0]), es_id_g);
#else
            ret = H5Dwrite(offsetdataset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, dxpl_ind, &(offsets[0]));
#endif
            if(ret < 0) { std::cout << "Write offset dataset failed! ret = " << ret << std::endl; }

#ifdef AMREX_USE_HDF5_ASYNC
            ret = H5Dwrite_async(centerdataset, center_id, H5S_ALL, H5S_ALL, dxpl_ind, &(centering[0]), es_id_g);
#else
            ret = H5Dwrite(centerdataset, center_id, H5S_ALL, H5S_ALL, dxpl_ind, &(centering[0]));
#endif
            if(ret < 0) { std::cout << "Write center dataset failed! ret = " << ret << std::endl; }

#ifdef AMREX_USE_HDF5_ASYNC
            ret = H5Dwrite_async(boxdataset, babox_id, H5S_ALL, H5S_ALL, dxpl_ind, &(vbox[0]), es_id_g);
#else
            ret = H5Dwrite(boxdataset, babox_id, H5S_ALL, H5S_ALL, dxpl_ind, &(vbox[0]));
#endif
            if(ret < 0) { std::cout << "Write box dataset failed! ret = " << ret << std::endl; }
        } // end IOProcessor

        hsize_t hs_procsize[1], hs_allprocsize[1], ch_offset[1];

        ch_offset[0]       = procOffsets[myProc];          // ---- offset on this proc
        hs_procsize[0]     = procBufferSize[myProc];       // ---- size of buffer on this proc
        hs_allprocsize[0]  = offsets[sortedGrids.size()];  // ---- size of buffer on all procs

        hid_t dataspace    = H5Screate_simple(1, hs_allprocsize, NULL);
        hid_t memdataspace = H5Screate_simple(1, hs_procsize, NULL);

        if (hs_procsize[0] == 0)
            H5Sselect_none(dataspace);
        else
            H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, ch_offset, NULL, hs_procsize, NULL);

        Vector<Real> a_buffer(procBufferSize[myProc]*ncomp, -1.0);
        Vector<Real> a_buffer_ind(procBufferSize[myProc], -1.0);
        const MultiFab* data;
        std::unique_ptr<MultiFab> mf_tmp;
        if (mf[level]->nGrowVect() != 0) {
            mf_tmp = std::make_unique<MultiFab>(mf[level]->boxArray(),
                                                mf[level]->DistributionMap(),
                                                mf[level]->nComp(), 0, MFInfo(),
                                                mf[level]->Factory());
            MultiFab::Copy(*mf_tmp, *mf[level], 0, 0, mf[level]->nComp(), 0);
            data = mf_tmp.get();
        } else {
            data = mf[level];
        }

        hid_t dataset;
        char dataname[64];

#ifdef AMREX_USE_HDF5_SZ
        if (mode_env == "SZ") {
            size_t cd_nelmts;
            unsigned int* cd_values = NULL;
            unsigned filter_config;
            SZ_metaDataToCdArray(&cd_nelmts, &cd_values, SZ_DOUBLE, 0, 0, 0, 0, hs_allprocsize[0]);
            H5Pset_filter(dcpl_id, H5Z_FILTER_SZ, H5Z_FLAG_MANDATORY, cd_nelmts, cd_values);
        }
#endif

        BL_PROFILE_VAR("H5DwriteData", h5dwg);

        for (int jj = 0; jj < ncomp; jj++) {

            Long writeDataItems(0), writeDataSize(0);
            for(MFIter mfi(*data); mfi.isValid(); ++mfi) {
                const FArrayBox &fab = (*data)[mfi];
                writeDataItems = fab.box().numPts();
                if(doConvert) {
                    RealDescriptor::convertFromNativeFormat(static_cast<void *> (a_buffer.dataPtr()),
                                                            writeDataItems * ncomp, fab.dataPtr(), *whichRD);

                } else {    // ---- copy from the fab
                    memcpy(static_cast<void *> (a_buffer.dataPtr()),
                           fab.dataPtr(), writeDataItems * ncomp * whichRDBytes);
                }

                // Extract individual variable data
                memcpy(static_cast<void *> (a_buffer_ind.dataPtr() + writeDataSize),
                       static_cast<void *> (a_buffer.dataPtr() + jj*writeDataItems),
                       writeDataItems * whichRDBytes);

                writeDataSize += writeDataItems;
            }

            sprintf(dataname, "data:datatype=%d", jj);
#ifdef AMREX_USE_HDF5_ASYNC
            dataset = H5Dcreate_async(grp, dataname, H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, dcpl_id, H5P_DEFAULT, es_id_g);
            if(dataset < 0) std::cout << ParallelDescriptor::MyProc() << "create data failed!  ret = " << dataset << std::endl;
            ret = H5Dwrite_async(dataset, H5T_NATIVE_DOUBLE, memdataspace, dataspace, dxpl_col, a_buffer_ind.dataPtr(), es_id_g);
            if(ret < 0) { std::cout << ParallelDescriptor::MyProc() << "Write data failed!  ret = " << ret << std::endl; break; }
            H5Dclose_async(dataset, es_id_g);
#else
            dataset = H5Dcreate(grp, dataname, H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
            if(dataset < 0) std::cout << ParallelDescriptor::MyProc() << "create data failed!  ret = " << dataset << std::endl;
            ret = H5Dwrite(dataset, H5T_NATIVE_DOUBLE, memdataspace, dataspace, dxpl_col, a_buffer_ind.dataPtr());
            if(ret < 0) { std::cout << ParallelDescriptor::MyProc() << "Write data failed!  ret = " << ret << std::endl; break; }
            H5Dclose(dataset);
#endif
        }

        BL_PROFILE_VAR_STOP(h5dwg);

        H5Sclose(memdataspace);
        H5Sclose(dataspace);
        H5Sclose(offsetdataspace);
        H5Sclose(centerdataspace);
        H5Sclose(boxdataspace);

#ifdef AMREX_USE_HDF5_ASYNC
        H5Dclose_async(offsetdataset, es_id_g);
        H5Dclose_async(centerdataset, es_id_g);
        H5Dclose_async(boxdataset, es_id_g);
        H5Gclose_async(grp, es_id_g);
#else
        H5Dclose(offsetdataset);
        H5Dclose(centerdataset);
        H5Dclose(boxdataset);
        H5Gclose(grp);
#endif
    } // For group

    BL_PROFILE_VAR_STOP(h5dwd);

    H5Tclose(center_id);
    H5Tclose(babox_id);
    H5Pclose(fapl);
    H5Pclose(dcpl_id);
    H5Pclose(dxpl_col);
    H5Pclose(dxpl_ind);

#ifdef AMREX_USE_HDF5_ASYNC
    H5Fclose_async(fid, es_id_g);
#else
    H5Fclose(fid);
#endif
} // WriteMultiLevelPlotfileHDF5MultiDset

void
WriteSingleLevelPlotfileHDF5 (const std::string& plotfilename,
                              const MultiFab& mf, const Vector<std::string>& varnames,
                              const Geometry& geom, Real time, int level_step,
                              const std::string &compression,
                              const std::string &versionName,
                              const std::string &levelPrefix,
                              const std::string &mfPrefix,
                              const Vector<std::string>& extra_dirs)
{
    Vector<const MultiFab*> mfarr(1,&mf);
    Vector<Geometry> geomarr(1,geom);
    Vector<int> level_steps(1,level_step);
    Vector<IntVect> ref_ratio;

    WriteMultiLevelPlotfileHDF5(plotfilename, 1, mfarr, varnames, geomarr, time, level_steps, ref_ratio,
                                compression, versionName, levelPrefix, mfPrefix, extra_dirs);
}

void
WriteSingleLevelPlotfileHDF5SingleDset (const std::string& plotfilename,
                                        const MultiFab& mf, const Vector<std::string>& varnames,
                                        const Geometry& geom, Real time, int level_step,
                                        const std::string &compression,
                                        const std::string &versionName,
                                        const std::string &levelPrefix,
                                        const std::string &mfPrefix,
                                        const Vector<std::string>& extra_dirs)
{
    Vector<const MultiFab*> mfarr(1,&mf);
    Vector<Geometry> geomarr(1,geom);
    Vector<int> level_steps(1,level_step);
    Vector<IntVect> ref_ratio;

    WriteMultiLevelPlotfileHDF5SingleDset(plotfilename, 1, mfarr, varnames, geomarr, time, level_steps, ref_ratio,
                                          compression, versionName, levelPrefix, mfPrefix, extra_dirs);
}

void
WriteSingleLevelPlotfileHDF5MultiDset (const std::string& plotfilename,
                                       const MultiFab& mf, const Vector<std::string>& varnames,
                                       const Geometry& geom, Real time, int level_step,
                                       const std::string &compression,
                                       const std::string &versionName,
                                       const std::string &levelPrefix,
                                       const std::string &mfPrefix,
                                       const Vector<std::string>& extra_dirs)
{
    Vector<const MultiFab*> mfarr(1,&mf);
    Vector<Geometry> geomarr(1,geom);
    Vector<int> level_steps(1,level_step);
    Vector<IntVect> ref_ratio;

    WriteMultiLevelPlotfileHDF5MultiDset(plotfilename, 1, mfarr, varnames, geomarr, time, level_steps, ref_ratio,
                                         compression, versionName, levelPrefix, mfPrefix, extra_dirs);
}

void
WriteMultiLevelPlotfileHDF5 (const std::string &plotfilename,
                             int nlevels,
                             const Vector<const MultiFab*> &mf,
                             const Vector<std::string> &varnames,
                             const Vector<Geometry> &geom,
                             Real time,
                             const Vector<int> &level_steps,
                             const Vector<IntVect> &ref_ratio,
                             const std::string &compression,
                             const std::string &versionName,
                             const std::string &levelPrefix,
                             const std::string &mfPrefix,
                             const Vector<std::string>& extra_dirs)
{

    WriteMultiLevelPlotfileHDF5SingleDset(plotfilename, nlevels, mf, varnames, geom, time, level_steps, ref_ratio,
                                          compression, versionName, levelPrefix, mfPrefix, extra_dirs);
}

void
WriteSingleLevelPlotfileHDF5MD (const std::string& plotfilename,
                                const MultiFab& mf, const Vector<std::string>& varnames,
                                const Geometry& geom, Real time, int level_step,
                                const std::string &compression,
                                const std::string &versionName,
                                const std::string &levelPrefix,
                                const std::string &mfPrefix,
                                const Vector<std::string>& extra_dirs)
{
    Vector<const MultiFab*> mfarr(1,&mf);
    Vector<Geometry> geomarr(1,geom);
    Vector<int> level_steps(1,level_step);
    Vector<IntVect> ref_ratio;

    WriteMultiLevelPlotfileHDF5MD(plotfilename, 1, mfarr, varnames, geomarr, time, level_steps, ref_ratio,
                                  compression, versionName, levelPrefix, mfPrefix, extra_dirs);
}

void WriteMultiLevelPlotfileHDF5MD (const std::string& plotfilename,
                                    int nlevels,
                                    const Vector<const MultiFab*>& mf,
                                    const Vector<std::string>& varnames,
                                    const Vector<Geometry>& geom,
                                    Real time,
                                    const Vector<int>& level_steps,
                                    const Vector<IntVect>& ref_ratio,
                                    const std::string &compression,
                                    const std::string &versionName,
                                    const std::string &levelPrefix,
                                    const std::string &mfPrefix,
                                    const Vector<std::string>& extra_dirs)
{
    BL_PROFILE("WriteMultiLevelPlotfileHDF5MultiDset");

    BL_ASSERT(nlevels <= mf.size());
    BL_ASSERT(nlevels <= geom.size());
    BL_ASSERT(nlevels <= ref_ratio.size()+1);
    BL_ASSERT(nlevels <= level_steps.size());
    BL_ASSERT(mf[0]->nComp() == varnames.size());

    int myProc(ParallelDescriptor::MyProc());
    int nProcs(ParallelDescriptor::NProcs());
    int finestLevel(nlevels-1);
    int nComp(mf[0]->nComp());
    int level(0);
    herr_t ret(0);
    std::string filename(plotfilename + ".h5");

    auto whichRD = FArrayBox::getDataDescriptor();
    bool doConvert(*whichRD != FPC::NativeRealDescriptor());
    int whichRDBytes(whichRD->numBytes());
    hsize_t chunkDims[AMREX_SPACEDIM];

#ifdef AMREX_USE_HDF5_ASYNC
    // For HDF5 async VOL, block and wait previous tasks have all completed
    if (es_id_g != 0) {
        async_vol_es_wait();
    }
    else {
        ExecOnFinalize(async_vol_es_wait_close);
        es_id_g = H5EScreate();
    }
#endif

    hid_t fapl, dxpl_col, dxpl_ind, fid, grp, dcpl, fspace, dset, dspace;
    fapl     = H5Pcreate (H5P_FILE_ACCESS);
    dcpl     = H5Pcreate(H5P_DATASET_CREATE);
    dxpl_col = H5Pcreate(H5P_DATASET_XFER);
    dxpl_ind = H5Pcreate(H5P_DATASET_XFER);

    // Get domain
    hsize_t domain[AMREX_SPACEDIM*2], dims[5] = {0, 0, 0, 0, 0};
    int domain_shape[AMREX_SPACEDIM], centering;
    double domain_size[AMREX_SPACEDIM];
    Box tmp(geom[level].Domain());
    std::string levelName;
    std::string dsetName;
    BL_ASSERT(AMREX_SPACEDIM <= 5);
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        centering = tmp.ixType().test(i) ? 1 : 0;
        domain[i] = (hsize_t)tmp.smallEnd(i);
        domain[i+AMREX_SPACEDIM] = (hsize_t)tmp.bigEnd(i);
        domain_shape[i] = (int)(tmp.bigEnd(i) - tmp.smallEnd(i) + 1 - centering);
        dims[i] = (hsize_t)(tmp.bigEnd(i) - tmp.smallEnd(i) + 1 - centering);
        if (domain[i] != 0 && myProc == ParallelDescriptor::IOProcessor())
            std::cerr << "Domain lo has non-zero dim!" << i << domain[i] << std::endl;
        domain_size[i] = geom[level].ProbHi(i);
    }

    // Write out root level metadata, 1 rank create file and write metadata, no async
    if(ParallelDescriptor::IOProcessor()) {

        fid = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (fid < 0)
            FileOpenFailed(filename.c_str());

        // Level 0, create and write entire domain
        level = 0;

        CreateWriteHDF5AttrString(fid, "version", versionName.c_str());

        CreateWriteHDF5AttrString(fid, "format", "AMReX_HDF5_Multi_Dim_by_Level_Forder");

        grp = H5Gcreate(fid, "domain", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (grp < 0)
            amrex::Abort("Domain group create failed!");

        CreateWriteHDF5AttrInt(grp, "shape", AMREX_SPACEDIM, domain_shape);
        CreateWriteHDF5AttrDouble(grp, "size", AMREX_SPACEDIM, domain_size);

        int type[AMREX_SPACEDIM];
        for (int i = 0; i < AMREX_SPACEDIM; ++i)
            type[i] = (int)geom[level].Domain().ixType().test(i) ? 1 : 0;

        CreateWriteHDF5AttrInt(grp, "domain_type", AMREX_SPACEDIM, type);

        CreateWriteHDF5AttrInt(grp, "steps", 1, &level_steps[level]);

        H5Gclose(grp);

        // level 0
        levelName = "level_" + std::to_string(level);
        grp = H5Gcreate(fid, levelName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (grp < 0)
            amrex::Abort("Level group create failed!");

        int nbox = mf[level]->boxArray().size();
        CreateWriteHDF5AttrInt(grp, "nbox", 1, &nbox);

        double cur_time = (double)time;
        CreateWriteHDF5AttrDouble(grp, "time", 1, &cur_time);

        int ngrow = mf[level]->nGrow();
        CreateWriteHDF5AttrInt(grp, "ngrow", 1, &ngrow);

        H5Gclose(grp);

        for (int i = 1; i < nlevels; i++) {
            levelName = "level_" + std::to_string(i);
            grp = H5Gcreate(fid, levelName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if (grp < 0)
                amrex::Abort("Level group create failed!");
            H5Gclose(grp);
        }

        H5Fclose(fid);
    } // End if IOProcessor

    // Need a barrier so other ranks open after file is created
    ParallelDescriptor::Barrier();

#ifdef BL_USE_MPI
    SetHDF5fapl(fapl, ParallelDescriptor::Communicator());
    H5Pset_dxpl_mpio(dxpl_col, H5FD_MPIO_COLLECTIVE);
#else
    SetHDF5fapl(fapl);
#endif

    // All process open the file
#ifdef AMREX_USE_HDF5_ASYNC
    // Only use async for writing actual data
    fid = H5Fopen_async(filename.c_str(), H5F_ACC_RDWR, fapl, es_id_g);
#else
    fid = H5Fopen(filename.c_str(), H5F_ACC_RDWR, fapl);
#endif
    if (fid < 0)
        FileOpenFailed(filename.c_str());

    // level 0
    level = 0;
    levelName = "level_" + std::to_string(level);

#ifdef AMREX_USE_HDF5_ASYNC
    grp = H5Gopen_async(fid, levelName.c_str(), H5P_DEFAULT, es_id_g);
#else
    grp = H5Gopen(fid, levelName.c_str(), H5P_DEFAULT);
#endif
    if (grp < 0) amrex::Abort("Level group open failed!");

    const MultiFab* mfData0;
    std::unique_ptr<MultiFab> mf_tmp1;
    if (mf[level]->nGrowVect() != 0) {
	mf_tmp1 = std::make_unique<MultiFab>(mf[level]->boxArray(),
					    mf[level]->DistributionMap(),
					    mf[level]->nComp(), 0, MFInfo(),
					    mf[level]->Factory());
	MultiFab::Copy(*mf_tmp1, *mf[level], 0, 0, mf[level]->nComp(), 0);
	mfData0 = mf_tmp1.get();
    } else {
	mfData0 = mf[level];
    }


    // Find out the minimal box size to use for chunk size
    Vector<int> procMap0 = mf[level]->DistributionMap().ProcessorMap();
    Vector<int> procBoxCount0(nProcs, 0);
    int maxProcBoxCount(0), gridProc(0), maxProcNpts(0);
    const BoxArray& grids0 = mf[level]->boxArray();
    maxProcNpts = grids0[0].numPts();
    for (int j = 0; j < AMREX_SPACEDIM; j++) {
        centering = grids0[0].ixType().test(j) ? 1 : 0;
        chunkDims[j] = grids0[0].bigEnd(j) - grids0[0].smallEnd(j) + 1 - centering;
    }

    for(int i = 0; i < grids0.size(); i++) {
        if (grids0[i].numPts() < maxProcNpts) {
            maxProcNpts = grids0[i].numPts();
            for (int j = 0; j < AMREX_SPACEDIM; j++) {
                centering = grids0[i].ixType().test(j) ? 1 : 0;
                chunkDims[j] = grids0[i].bigEnd(j) - grids0[i].smallEnd(j) + 1 - centering;
            }
        }
        gridProc = procMap0[i];
        procBoxCount0[gridProc]++;
        if (procBoxCount0[gridProc] > maxProcBoxCount)
            maxProcBoxCount = procBoxCount0[gridProc];
        /* // Debug */
        /* if (myProc == 0) { */
        /*     fprintf(stderr, "Box %d assigned to rank %d, current maxProcBoxCount %d, chunk %llu %llu %llu\n", */ 
        /*             i, gridProc, maxProcBoxCount, chunkDims[0], chunkDims[1], chunkDims[2]); */
        /* } */
    }

    H5Pset_chunk(dcpl, AMREX_SPACEDIM, chunkDims);
    H5Pset_fill_time(dcpl, H5D_FILL_TIME_NEVER);
    H5Pset_alloc_time(dcpl, H5D_ALLOC_TIME_LATE);

#if (defined AMREX_USE_HDF5_ZFP) || (defined AMREX_USE_HDF5_SZ)
    std::string::size_type pos = compression.find('@');
    if (pos != std::string::npos) {
        mode_env = compression.substr(0, pos);
        value_env = compression.substr(pos+1);
        if (!value_env.empty()) {
            comp_value = atof(value_env.c_str());
        }
    }

#ifdef AMREX_USE_HDF5_ZFP
    pos = compression.find("ZFP");
    if (pos != std::string::npos) {
        ret = H5Z_zfp_initialize();
        if (ret < 0) amrex::Abort("ZFP initialize failed!");
    }
#endif

#ifdef AMREX_USE_HDF5_SZ
    pos = compression.find("SZ");
    if (pos != std::string::npos) {
        ret = H5Z_SZ_Init((char*)value_env.c_str());
        if (ret < 0) {
            std::cout << "SZ config file:" << value_env.c_str() << std::endl;
            amrex::Abort("SZ initialize failed, check SZ config file!");
        }
    }
#endif

    if (!mode_env.empty() && mode_env != "None") {
        if (mode_env == "ZLIB")
            H5Pset_deflate(dcpl, (int)comp_value);
#ifdef AMREX_USE_HDF5_ZFP
        else if (mode_env == "ZFP_RATE")
            H5Pset_zfp_rate(dcpl, comp_value);
        else if (mode_env == "ZFP_PRECISION")
            H5Pset_zfp_precision(dcpl, (unsigned int)comp_value);
        else if (mode_env == "ZFP_ACCURACY")
            H5Pset_zfp_accuracy(dcpl, comp_value);
        else if (mode_env == "ZFP_REVERSIBLE")
            H5Pset_zfp_reversible(dcpl);
        else if (mode_env == "ZLIB")
            H5Pset_deflate(dcpl, (int)comp_value);
#endif

        if (ParallelDescriptor::MyProc() == 0) {
            std::cout << "\nHDF5 plotfile using " << mode_env << ", " <<
                value_env << ", " << chunkDims << std::endl;
        }
    }
#endif

    hid_t* dsetIds = new hid_t[nComp]();
    fspace = H5Screate_simple(AMREX_SPACEDIM, dims, NULL);

#ifdef AMREX_USE_HDF5_SZ
    if (mode_env == "SZ") {
        size_t cd_nelmts;
        unsigned int* cd_values = NULL;
        unsigned filter_config;
        SZ_metaDataToCdArray(&cd_nelmts, &cd_values, SZ_DOUBLE, dims[4], dims[3], dims[2], dims[1], dims[0]);
        H5Pset_filter(dcpl, H5Z_FILTER_SZ, H5Z_FLAG_MANDATORY, cd_nelmts, cd_values);
    }
#endif

    for (int i = 0; i < nComp; i++) {
#ifdef AMREX_USE_HDF5_ASYNC
        dsetIds[i] = H5Dcreate_async(grp, varnames[i].c_str(), H5T_NATIVE_DOUBLE, fspace, 
                               H5P_DEFAULT, dcpl, H5P_DEFAULT, es_id_g);
        if(dsetIds[i] < 0) amrex::Abort("H5Dcreate_async failed!");
#else
        dsetIds[i] = H5Dcreate(grp, varnames[i].c_str(), H5T_NATIVE_DOUBLE, fspace, 
                               H5P_DEFAULT, dcpl, H5P_DEFAULT);
        if(dsetIds[i] < 0) amrex::Abort("H5Dcreate failed!");
#endif
    }

    hsize_t myOffset[AMREX_SPACEDIM], myCount[AMREX_SPACEDIM];
    hsize_t boxOffset[AMREX_SPACEDIM], boxCount[AMREX_SPACEDIM];
    hsize_t fileOffset[AMREX_SPACEDIM+1], fileCount[AMREX_SPACEDIM+1];
    int myBoxCount(0);
    // Iterate over boxes and write data
    for(MFIter mfi(*mfData0); mfi.isValid(); ++mfi) {
        myBoxCount++;
        const FArrayBox &fab = (*mfData0)[mfi];
        Long writeDataItems = fab.box().numPts() * nComp;
        Long writeDataSize = writeDataItems * whichRDBytes;
        hsize_t varDataSize = fab.box().numPts() * whichRDBytes;
        char *dataPtr = new char[writeDataSize];
        
        for (int j = 0; j < AMREX_SPACEDIM; j++) {
            centering = fab.box().ixType().test(j) ? 1 : 0;
            myOffset[j] = fab.box().smallEnd(j);
            myCount[j]  = fab.box().bigEnd(j) - myOffset[j] + 1 - centering;
        }

        hid_t dspace = H5Screate_simple(AMREX_SPACEDIM, myCount, NULL);

        H5Sselect_hyperslab(fspace, H5S_SELECT_SET, myOffset, NULL, myCount, NULL);

        // Get data pointer
        Real const* fabdata = fab.dataPtr();
#ifdef AMREX_USE_GPU
        std::unique_ptr<FArrayBox> hostfab;
        if (fab.arena()->isManaged() || fab.arena()->isDevice()) {
            hostfab = std::make_unique<FArrayBox>(fab.box(), nComp,
                                                  The_Pinned_Arena());
            Gpu::dtoh_memcpy_async(hostfab->dataPtr(), fab.dataPtr(),
                                   fab.size()*sizeof(Real));
            Gpu::streamSynchronize();
            fabdata = hostfab->dataPtr();
        }
#endif
        if(doConvert) {
            RealDescriptor::convertFromNativeFormat(dataPtr,
                                                    writeDataItems,
                                                    fabdata, *whichRD);
        } else {    // ---- copy from the fab
            memcpy(dataPtr, fabdata, writeDataSize);
        }

        for (int j = 0; j < nComp; j++) {
#ifdef AMREX_USE_HDF5_ASYNC
            ret = H5Dwrite_async(dsetIds[j], H5T_NATIVE_DOUBLE, dspace, fspace, dxpl_col,
                                 &(dataPtr[j*varDataSize]), es_id_g);
            if(ret < 0) amrex::Abort("H5Dwrite_async failed!");
#else
            ret = H5Dwrite(dsetIds[j], H5T_NATIVE_DOUBLE, dspace, fspace, dxpl_col,
                           &(dataPtr[j*varDataSize]));
            if(ret < 0) amrex::Abort("H5Dwrite failed!");
#endif
        } // End for comp

        /* fprintf(stderr, "Rank %d: %d %d %d, %d %d %d, written %d\n", myProc, */
        /*         fab.box().smallEnd(0), fab.box().smallEnd(1), fab.box().smallEnd(2), */ 
        /*         fab.box().bigEnd(0), fab.box().bigEnd(1), fab.box().bigEnd(2), myBoxCount); */

        H5Sclose(dspace);
        delete [] dataPtr;
    }

    // Residue write if less boxes is written than others due to chunking and collective I/O
    for (int i = myBoxCount; i < maxProcBoxCount; i++) {
        H5Sselect_none(fspace);
        for (int j = 0; j < nComp; j++) {
#ifdef AMREX_USE_HDF5_ASYNC
            ret = H5Dwrite_async(dsetIds[j], H5T_NATIVE_DOUBLE, H5S_ALL, fspace, dxpl_col,
                                 NULL, es_id_g);
            if(ret < 0) amrex::Abort("H5Dwrite_async failed!");
#else
            ret = H5Dwrite(dsetIds[j], H5T_NATIVE_DOUBLE, H5S_ALL, fspace, dxpl_col, NULL);
            if(ret < 0) amrex::Abort("H5Dwrite failed!");
#endif
        } // End for comp
    }
    H5Sclose(fspace);

#ifdef AMREX_USE_HDF5_ASYNC
    for (int i = 0; i < nComp; i++) {
        ret = H5Dclose_async(dsetIds[i], es_id_g);
        if(ret < 0) amrex::Abort("H5Dclose_async failed!");
    }
    ret = H5Gclose_async(grp, es_id_g);
    if(ret < 0) amrex::Abort("H5Gclose_async failed!");
#else
    for (int i = 0; i < nComp; i++) {
        ret = H5Dclose(dsetIds[i]);
        if(ret < 0) amrex::Abort("H5Dclose failed!");
    }
    ret = H5Gclose(grp);
    if(ret < 0) amrex::Abort("H5Gclose failed!");
#endif
    // End of level 0

    /* fprintf(stderr, "Rank %d: finished level 0\n", myProc); */

    // Upper levels
    for (level = 1; level <= finestLevel; level++) {
        levelName = "level_" + std::to_string(level);

#ifdef AMREX_USE_HDF5_ASYNC
        grp = H5Gopen_async(fid, levelName.c_str(), H5P_DEFAULT, es_id_g);
#else
        grp = H5Gopen(fid, levelName.c_str(), H5P_DEFAULT);
#endif
        if (grp < 0) amrex::Abort("Level group open failed!");

        const MultiFab* mfData1;
        std::unique_ptr<MultiFab> mf_tmp0;
        if (mf[level]->nGrowVect() != 0) {
            mf_tmp0 = std::make_unique<MultiFab>(mf[level]->boxArray(),
                                                mf[level]->DistributionMap(),
                                                mf[level]->nComp(), 0, MFInfo(),
                                                mf[level]->Factory());
            MultiFab::Copy(*mf_tmp0, *mf[level], 0, 0, mf[level]->nComp(), 0);
            mfData1 = mf_tmp0.get();
        } else {
            mfData1 = mf[level];
        }

        const BoxArray& grids1 = mf[level]->boxArray();
        Vector<int> procMap1 = mf[level]->DistributionMap().ProcessorMap();

        // boxSizeCountID value, first is count, second is corresponding ID used in dset name
        std::map<std::vector<int>, std::vector<int>> boxSizeCountID;
        std::map<std::vector<int>, std::vector<int>>::iterator boxSizeCountIt;

        std::vector<std::vector<int>> boxSmallEndBySize;
        std::vector<std::vector<int>> boxBigEndBySize;

        int boxSizeID = 0;
        for(int i = 0; i < grids1.size(); i++) {

            std::vector<int> myBoxSize(AMREX_SPACEDIM);
            // Get the small and big end of current boxes
            for (int j = 0; j < AMREX_SPACEDIM; j++) {
                centering = grids1[i].ixType().test(j) ? 1 : 0;
                myBoxSize[j] = grids1[i].bigEnd(j) - grids1[i].smallEnd(j) + 1 - centering;
            }

            boxSizeCountIt = boxSizeCountID.find(myBoxSize);
            if (boxSizeCountIt == boxSizeCountID.end()) {
                // Current box has a new box size
                std::vector<int> myBoxValue(2);
                myBoxValue[0] = 1;
                myBoxValue[1] = boxSizeID;
                boxSizeCountID.insert(std::pair<std::vector<int>, std::vector<int>>(myBoxSize, myBoxValue));

                // Allocate the vector to store all small and big ends
                std::vector<int> boxSmallEnd(AMREX_SPACEDIM*grids1.size(), 0);
                std::vector<int> boxBigEnd(AMREX_SPACEDIM*grids1.size(), 0);
                for (int j = 0; j < AMREX_SPACEDIM; j++) {
                    boxSmallEnd[j] = grids1[i].smallEnd(j);
                    boxBigEnd[j] = grids1[i].bigEnd(j);
                }
                boxSmallEndBySize.push_back(boxSmallEnd);
                boxBigEndBySize.push_back(boxBigEnd);
                boxSizeID++;
            }
            else {
                 int prevCnt = boxSizeCountIt->second[0];
                 int myID = boxSizeCountIt->second[1];
                 boxSizeCountIt->second[0] = prevCnt + 1;
                 for (int j = 0; j < AMREX_SPACEDIM; j++) {
                     boxSmallEndBySize[myID][prevCnt*AMREX_SPACEDIM+j] = grids1[i].smallEnd(j);
                     boxBigEndBySize[myID][prevCnt*AMREX_SPACEDIM+j] = grids1[i].bigEnd(j);
                 }
            }
        }

        /* fprintf(stderr, "Rank %d: map has %d items, last one has %d count\n", myProc, boxSizeCountID.size(), boxSizeCountIt->second[0]); */

        // Write out small end and big end for each box
        for (boxSizeCountIt = boxSizeCountID.begin(); boxSizeCountIt != boxSizeCountID.end(); ++boxSizeCountIt) {
            int nbox = boxSizeCountIt->second[0];
            boxSizeID = boxSizeCountIt->second[1];
            hsize_t boxSpace[2];
            boxSpace[0] = (hsize_t)nbox;
            boxSpace[1] = (hsize_t)AMREX_SPACEDIM;

            // Small end dset
            fspace = H5Screate_simple(2, boxSpace, NULL);
            dsetName = "small_end#" + std::to_string(boxSizeID);
#ifdef AMREX_USE_HDF5_ASYNC
            dset = H5Dcreate_async(grp, dsetName.c_str(), H5T_NATIVE_INT, fspace, 
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT, es_id_g);
            if(dset < 0) amrex::Abort("H5Dcreate_async failed!");

            ret = H5Dwrite_async(dset, H5T_NATIVE_INT, H5S_ALL, fspace, dxpl_col,
                                 &boxSmallEndBySize[boxSizeID][0], es_id_g);
            if(ret < 0) amrex::Abort("H5Dwrite_async failed!");

            ret = H5Dclose_async(dset, es_id_g);
            if(ret < 0) amrex::Abort("H5Dclose_async failed!");
#else
            dset = H5Dcreate(grp, dsetName.c_str(), H5T_NATIVE_INT, fspace, 
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if(dset < 0) amrex::Abort("H5Dcreate failed!");

            ret = H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, fspace, dxpl_col, 
                           &boxSmallEndBySize[boxSizeID][0]);
            if(ret < 0) amrex::Abort("H5Dwrite failed!");

            ret = H5Dclose(dset);
            if(ret < 0) amrex::Abort("H5Dclose failed!");
#endif
            H5Sclose(fspace);

            int ngrow = mf[level]->nGrow();
            double cur_time = (double)time;
            int ratio = 1;
            if (ref_ratio.size() > 0)
                ratio = ref_ratio[level][0];
            // Big end dset
            fspace = H5Screate_simple(2, boxSpace, NULL);
            dsetName = "big_end#" + std::to_string(boxSizeID);
#ifdef AMREX_USE_HDF5_ASYNC
            dset = H5Dcreate_async(grp, dsetName.c_str(), H5T_NATIVE_INT, fspace, 
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT, es_id_g);
            if(dset < 0) amrex::Abort("H5Dcreate_async failed!");

            ret = H5Dwrite_async(dset, H5T_NATIVE_INT, H5S_ALL, fspace, dxpl_col,
                                 &boxBigEndBySize[boxSizeID][0], es_id_g);
            if(ret < 0) amrex::Abort("H5Dwrite_async failed!");

            ret = H5Dclose_async(dset, es_id_g);
            if(ret < 0) amrex::Abort("H5Dclose_async failed!");
            CreateWriteHDF5AttrIntAsync(grp, dsetName.c_str(), 1, &nbox);

            CreateWriteHDF5AttrIntAsync(grp, "ngrow", 1, &ngrow);

            CreateWriteHDF5AttrDoubleAsync(grp, "time", 1, &cur_time);

            CreateWriteHDF5AttrIntAsync(grp, "ref_ratio", 1, &ratio);
#else
            dset = H5Dcreate(grp, dsetName.c_str(), H5T_NATIVE_INT, fspace, 
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if(dset < 0) amrex::Abort("H5Dcreate failed!");

            ret = H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, fspace, dxpl_col, 
                           &boxBigEndBySize[boxSizeID][0]);
            if(ret < 0) amrex::Abort("H5Dwrite failed!");

            ret = H5Dclose(dset);
            if(ret < 0) amrex::Abort("H5Dclose failed!");

            dsetName = "nbox#" + std::to_string(boxSizeID);
            CreateWriteHDF5AttrInt(grp, dsetName.c_str(), 1, &nbox);

            CreateWriteHDF5AttrInt(grp, "ngrow", 1, &ngrow);

            CreateWriteHDF5AttrDouble(grp, "time", 1, &cur_time);

            CreateWriteHDF5AttrInt(grp, "ref_ratio", 1, &ratio);
#endif
            H5Sclose(fspace);

        } // End iter boxSizeCountID write small and big ends

        // Iterate over all box sizes and create dsets
        for (boxSizeCountIt=boxSizeCountID.begin(); boxSizeCountIt != boxSizeCountID.end(); ++boxSizeCountIt) {
            // Get the box size from the map
            for (int j = 0; j < AMREX_SPACEDIM; j++)
                fileCount[j+1]  = boxSizeCountIt->first[j];

            // Expand the first dimension so we can write multiple same-size boxes to this dset
            fileCount[0] = boxSizeCountIt->second[0];
            fspace = H5Screate_simple(AMREX_SPACEDIM+1, fileCount, NULL);

            boxSizeID = boxSizeCountIt->second[1];

            for (int j = 0; j < nComp; j++) {
                dsetName = varnames[j] + "#" + std::to_string(boxSizeID);
#ifdef AMREX_USE_HDF5_ASYNC
                dset = H5Dcreate_async(grp, dsetName.c_str(), H5T_NATIVE_DOUBLE, fspace, 
                                       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT, es_id_g);
                if(dset < 0) amrex::Abort("H5Dcreate_async failed!");

                ret = H5Dclose_async(dset, es_id_g);
                if(ret < 0) amrex::Abort("H5Dclose_async failed 2!");
#else
                dset = H5Dcreate(grp, dsetName.c_str(), H5T_NATIVE_DOUBLE, fspace, 
                                       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                if(dset < 0) amrex::Abort("H5Dcreate failed!");

                ret = H5Dclose(dset);
                if(ret < 0) amrex::Abort("H5Dclose failed 2!");
#endif
            }
        } // End iter box sizes

        std::vector<int> hasWrittenSizeID(boxSizeCountID.size(), 0);
 
        // Iterate over all boxes and write to corresponding aggregated dset
        for(int i = 0; i < grids1.size(); i++) {
            int nbox;
            gridProc = procMap1[i];
            std::vector<int> myBoxSize(AMREX_SPACEDIM);

            // Get the small and big end of current boxes
            for (int j = 0; j < AMREX_SPACEDIM; j++) {
                centering = grids1[i].ixType().test(j) ? 1 : 0;
                boxOffset[j] = grids1[i].smallEnd(j);
                boxCount[j]  = grids1[i].bigEnd(j) - boxOffset[j] + 1 - centering;
                fileCount[j+1] = boxCount[j];
                myBoxSize[j] = grids1[i].bigEnd(j) - grids1[i].smallEnd(j) + 1 - centering;
            }

            boxSizeCountIt = boxSizeCountID.find(myBoxSize);
            if (boxSizeCountIt != boxSizeCountID.end()) {
                nbox = boxSizeCountIt->second[0];
                boxSizeID = boxSizeCountIt->second[1];
            }
            else {
                amrex::Abort("Cannot find box in box size map!");
            }

            for (int j = 0; j < nComp; j++) {
                dsetName = varnames[j] + "#" + std::to_string(boxSizeID);
#ifdef AMREX_USE_HDF5_ASYNC
                dsetIds[j] = H5Dopen_async(grp, dsetName.c_str(), H5P_DEFAULT, es_id_g);
                if(dsetIds[j] < 0) amrex::Abort("H5Dopen_async failed!");
#else
                dsetIds[j] = H5Dopen(grp, dsetName.c_str(), H5P_DEFAULT);
                if(dsetIds[j] < 0) amrex::Abort("H5Dopen failed!");
#endif
            }

            dspace = H5Screate_simple(AMREX_SPACEDIM, boxCount, NULL);

            // Re-create the file space for this box size
            fileCount[0] = boxSizeCountIt->second[0];
            fspace = H5Screate_simple(AMREX_SPACEDIM+1, fileCount, NULL);
            // Find offset of current box based on the number of written boxes, and do hyperslab sel
            for (int j = 0; j < AMREX_SPACEDIM+1; j++)
                fileOffset[j] = 0;
            fileOffset[0] = hasWrittenSizeID[boxSizeID];
            hasWrittenSizeID[boxSizeID]++;

            /* fprintf(stderr, "Rank %d, box %d, start %d %d %d, count %d %d %d, file offset %d %d %d %d, count %d %d %d %d,\n", */
            /*         myProc, i, boxOffset[0], boxOffset[1], boxOffset[2], boxCount[0], boxCount[1], boxCount[2], */
            /*         fileOffset[0], fileOffset[1], fileOffset[2], fileOffset[3], fileCount[0], fileCount[1], fileCount[2], fileCount[3]); */

            fileCount[0] = 1;
            H5Sselect_hyperslab(fspace, H5S_SELECT_SET, fileOffset, NULL, fileCount, NULL);

            myBoxCount = 0;
            // Find data pointer of this box by iterate over all valid boxes on current rank
            int hasBox = 0;
            for(MFIter mfi(*mfData1); mfi.isValid(); ++mfi) {
                myBoxCount++;
                const FArrayBox &fab = (*mfData1)[mfi];
                Long writeDataItems = fab.box().numPts() * nComp;
                Long writeDataSize = writeDataItems * whichRDBytes;
                hsize_t varDataSize = fab.box().numPts() * whichRDBytes;
                
                // Find data ptr for current box
                int flag = 1;
                for (int j = 0; j < AMREX_SPACEDIM; j++) {
                    centering = fab.box().ixType().test(j) ? 1 : 0;
                    myOffset[j] = 0;
                    myCount[j]  = fab.box().bigEnd(j) - myOffset[j] + 1 - centering;

                    if (fab.box().smallEnd(j) != boxOffset[j]) {
                        flag = 0;
                        break;
                    }
                }
                // Not match
                if (flag == 0)
                    continue;

                hasBox = 1;
                char *dataPtr = new char[writeDataSize];

                // Get data pointer
                Real const* fabdata = fab.dataPtr();
#ifdef AMREX_USE_GPU
                std::unique_ptr<FArrayBox> hostfab;
                if (fab.arena()->isManaged() || fab.arena()->isDevice()) {
                    hostfab = std::make_unique<FArrayBox>(fab.box(), nComp,
                                                          The_Pinned_Arena());
                    Gpu::dtoh_memcpy_async(hostfab->dataPtr(), fab.dataPtr(),
                                           fab.size()*sizeof(Real));
                    Gpu::streamSynchronize();
                    fabdata = hostfab->dataPtr();
                }
#endif
                if(doConvert) {
                    RealDescriptor::convertFromNativeFormat(dataPtr,
                                                            writeDataItems,
                                                            fabdata, *whichRD);
                } else {    // ---- copy from the fab
                    memcpy(dataPtr, fabdata, writeDataSize);
                }

                /* fprintf(stderr, "Rank %d, box %d, mem sel %llu, file sel %llu\n", myProc, i, H5Sget_select_npoints(dspace), H5Sget_select_npoints(fspace)); */

                for (int j = 0; j < nComp; j++) {
            
#ifdef AMREX_USE_HDF5_ASYNC
                    ret = H5Dwrite_async(dsetIds[j], H5T_NATIVE_DOUBLE, dspace, fspace, dxpl_ind,
                                         &(dataPtr[j*varDataSize]), es_id_g);
                    if(ret < 0) amrex::Abort("H5Dwrite_async failed! 690");
#else
                    ret = H5Dwrite(dsetIds[j], H5T_NATIVE_DOUBLE, dspace, fspace, dxpl_ind,
                                   &(dataPtr[j*varDataSize]));
                    if(ret < 0) amrex::Abort("H5Dwrite failed! 694");
#endif
                } // End for comp

                /* fprintf(stderr, "Rank %d: %d %d %d, %d %d %d, written %d\n", myProc, */
                /*         fab.box().smallEnd(0), fab.box().smallEnd(1), fab.box().smallEnd(2), */ 
                /*         fab.box().bigEnd(0), fab.box().bigEnd(1), fab.box().bigEnd(2), myBoxCount); */

                delete [] dataPtr;
            } // End mf iterator

            for (int j = 0; j < nComp; j++) {
#ifdef AMREX_USE_HDF5_ASYNC
                ret = H5Dclose_async(dsetIds[j], es_id_g);
                if(ret < 0) amrex::Abort("H5Dclose_async failed 2!");
#else
                ret = H5Dclose(dsetIds[j]);
                if(ret < 0) amrex::Abort("H5Dclose failed 2!");
#endif
            }
            H5Sclose(fspace);
            H5Sclose(dspace);

        } // End for box

#ifdef AMREX_USE_HDF5_ASYNC
        ret = H5Gclose_async(grp, es_id_g);
        if(ret < 0) amrex::Abort("H5Gclose_async failed!");
#else
        ret = H5Gclose(grp);
        if(ret < 0) amrex::Abort("H5Gclose failed!");
#endif
    } // End for upper level

    /* BL_PROFILE_VAR_STOP(h5dwd); */

    H5Pclose(fapl);
    H5Pclose(dcpl);
    H5Pclose(dxpl_col);
    H5Pclose(dxpl_ind);
#ifdef AMREX_USE_HDF5_ASYNC
    H5Fclose_async(fid, es_id_g);
#else
    H5Fclose(fid);
#endif

    delete [] dsetIds;

    return;
} // WriteMultiLevelPlotfileHDF5MultiDset
} // namespace amrex
