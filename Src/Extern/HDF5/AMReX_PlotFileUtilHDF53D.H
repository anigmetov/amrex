void WriteMultiLevelPlotfileHDF5ND (const std::string& plotfilename,
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
    hsize_t chunk_dim[AMREX_SPACEDIM];

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

    hid_t fapl, dxpl_col, dxpl_ind, fid, grp, dcpl;
    fapl     = H5Pcreate (H5P_FILE_ACCESS);
    dcpl     = H5Pcreate(H5P_DATASET_CREATE);
    dxpl_col = H5Pcreate(H5P_DATASET_XFER);
    dxpl_ind = H5Pcreate(H5P_DATASET_XFER);

    // Get domain
    hsize_t domain[AMREX_SPACEDIM*2], dims[5] = {0, 0, 0, 0, 0};
    int domain_shape[AMREX_SPACEDIM], centering;
    double domain_size[AMREX_SPACEDIM];
    Box tmp(geom[level].Domain());
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

        CreateWriteHDF5AttrString(fid, "format", "AMReX_HDF5_Multi_Dim_by_Level_FOrder");

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
            levelName = "level_" + std::to_string(level);
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
        chunk_dim[j] = grids0[j].bigEnd(j) - grids0[j].smallEnd(j) + 1 - centering;
    }

    for(int i = 0; i < grids0.size(); i++) {
        if (grids0[i].numPts() < maxProcNpts) {
            maxProcNpts = grids0[i].numPts();
            for (int j = 0; j < AMREX_SPACEDIM; j++) {
                centering = grids0[i].ixType().test(j) ? 1 : 0;
                chunk_dim[j] = grids0[i].bigEnd(j) - grids0[i].smallEnd(j) + 1 - centering;
            }
        }
        gridProc = procMap0[i];
        procBoxCount0[gridProc]++;
        if (procBoxCount0[gridProc] > maxProcBoxCount)
            maxProcBoxCount = procBoxCount0[gridProc];
        // Debug
        if (myProc == 0) {
            printf("Box %d assigned to rank %d, current maxProcBoxCount %d, chunk %llu %llu %llu\n", 
                    i, gridProc, maxProcBoxCount, chunk_dim[0], chunk_dim[1], chunk_dim[2]);
        }
    }

    H5Pset_chunk(dcpl, AMREX_SPACEDIM, chunk_dim);
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

    hid_t* dsetIds = new hid_t[nComp]();
    hid_t fspace = H5Screate_simple(AMREX_SPACEDIM, dims, NULL);

#ifdef AMREX_USE_HDF5_SZ
    if (mode_env == "SZ") {
        size_t cd_nelmts;
        unsigned int* cd_values = NULL;
        unsigned filter_config;
        SZ_metaDataToCdArray(&cd_nelmts, &cd_values, SZ_DOUBLE, dims[4], dims[3], dims[2], dims[1], dims[0]);
        H5Pset_filter(dcpl_id, H5Z_FILTER_SZ, H5Z_FLAG_MANDATORY, cd_nelmts, cd_values);
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

        fprintf(stderr, "Rank %d: %d %d %d, %d %d %d, written %d\n", myProc,
                fab.box().smallEnd(0), fab.box().smallEnd(1), fab.box().smallEnd(2), 
                fab.box().bigEnd(0), fab.box().bigEnd(1), fab.box().bigEnd(2), myBoxCount);

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
        if(ret < 0) amrex::Abort("H5Dclose_async failed!");
    }
    ret = H5Gclose(grp);
    if(ret < 0) amrex::Abort("H5Gclose_async failed!");
#endif
    // End of level 0


    // TODO Upper levels
    //

    for (level = 0; level <= finestLevel; level++) {
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

        // Find out the minimal box size to use for chunk size
        Vector<int> procMap1 = mf[level]->DistributionMap().ProcessorMap();
        Vector<int> procBoxCount1(nProcs, 0);
        maxProcBoxCount = 0;
        gridProc = 0;
        maxProcNpts = 0;
        const BoxArray& grids1 = mf[level]->boxArray();
        maxProcNpts = grids1[level].numPts();
        std::map<std::string, std::vector<int>> boxMap;

        for (int j = 0; j < AMREX_SPACEDIM; j++) {
            centering = grids1[level].ixType().test(j) ? 1 : 0;
            chunk_dim[j] = grids1[j].bigEnd(j) - grids1[j].smallEnd(j) + 1 - centering;
        }

        // Find and store the different box sizes
        for(int i = 0; i < grids1.size(); i++) {

            std::string boxSizeString;
            for (int j = 0; j < AMREX_SPACEDIM; j++) {
                centering = grids1[i].ixType().test(j) ? 1 : 0;
                boxSizeString += std::to_string(grids1[i].bigEnd(j) - grids1[i].smallEnd(j) + 1 - centering);
                boxSizeString += "_";
            }
            // Check if current box size already exist
            std::map<std::string, std::vector<int>>::iterator it = boxMap.find(boxSizeString);
            if (it == boxMap.end()) {
                std::vector<int> boxIDs;
                boxIDs.push_back(i);
                boxMap.insert(std::pair<std::string, std::vector<int>>(boxSizeString, boxIDs));
            }
            else {
                 it->second.push_back(i);
            }

            if (grids1[i].numPts() < maxProcNpts) {
                maxProcNpts = grids1[i].numPts();
                for (int j = 0; j < AMREX_SPACEDIM; j++) {
                    centering = grids1[i].ixType().test(j) ? 1 : 0;
                    chunk_dim[j] = grids1[i].bigEnd(j) - grids1[i].smallEnd(j) + 1 - centering;
                }
            }
            gridProc = procMap1[i];
            procBoxCount1[gridProc]++;
            if (procBoxCount1[gridProc] > maxProcBoxCount)
                maxProcBoxCount = procBoxCount1[gridProc];
            // Debug
            if (myProc == 0) {
                printf("Box %d assigned to rank %d, current maxProcBoxCount %d, chunk %llu %llu %llu\n", 
                        i, gridProc, maxProcBoxCount, chunk_dim[0], chunk_dim[1], chunk_dim[2]);
            }
        }

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
            if(ret < 0) amrex::Abort("H5Dclose_async failed!");
        }
        ret = H5Gclose(grp);
        if(ret < 0) amrex::Abort("H5Gclose_async failed!");
#endif
    } // End for level

    // End of upper levels

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


