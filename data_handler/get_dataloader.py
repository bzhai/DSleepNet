from data_handler.dis_train_norm_dataloader import get_unifactor_filter_regroup_loader, \
    get_multifactor_filter_regroup_loader


def build_dataloader(cfg, args, fold_num, train_shuffle=True):
    if args.loader_type == "regroup":
        if type(args.dis_type) == list and len(args.dis_type) > 1:
            train_loader, val_loader, test_loader = \
                get_multifactor_filter_regroup_loader(cfg=cfg, batch_size=args.batch_size, seq_len=args.seq_len,
                                                      num_classes=args.num_classes, dataset=args.dataset,
                                                      num_train=args.num_train, dis_type=args.dis_type,
                                                      train_test_group=args.train_test_group,
                                                      train_shuffle=train_shuffle, feature_type=args.feature_type,
                                                      mask_att=args.mask_att)
        else:
            train_loader, val_loader, test_loader = \
                get_unifactor_filter_regroup_loader(cfg=cfg, batch_size=args.batch_size,
                                                    seq_len=args.seq_len, num_classes=args.num_classes,
                                                    dataset=args.dataset, num_train=args.num_train,
                                                    dis_type=args.dis_type, train_test_group=args.train_test_group,
                                                    train_shuffle=train_shuffle, feature_type=args.feature_type)

    return train_loader, val_loader, test_loader

