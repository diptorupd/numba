#include "call_lowering.hpp"

namespace
{
llvm::StringRef extract_bound_func_name(llvm::StringRef name)
{
    assert(!name.empty());
    auto len = name.find(' ');
    return name.substr(0, len);
}
}

CallOpLowering::CallOpLowering(
    mlir::TypeConverter&, mlir::MLIRContext* context,
    CallOpLowering::resolver_t resolver):
    OpRewritePattern(context), resolver(resolver) {}

mlir::LogicalResult CallOpLowering::matchAndRewrite(plier::PyCallOp op, mlir::PatternRewriter& rewriter) const
{
    auto operands = op.getOperands();
    if (operands.empty())
    {
        return mlir::failure();
    }
    auto func_type = operands[0].getType();
    if (!func_type.isa<plier::PyType>())
    {
        return mlir::failure();
    }
    auto name = func_type.cast<plier::PyType>().getName();
    llvm::SmallVector<mlir::Type, 8> arg_types;
    llvm::SmallVector<mlir::Value, 8> args;
    if (name.consume_front("Function(") && name.consume_back(")"))
    {
        llvm::copy(llvm::drop_begin(op.getOperandTypes(), 1), std::back_inserter(arg_types));
        llvm::copy(llvm::drop_begin(op.getOperands(), 1), std::back_inserter(args));
        // TODO kwargs
    }
    else if (name.consume_front("BoundFunction(") && name.consume_back(")"))
    {
        auto getattr = mlir::dyn_cast<plier::GetattrOp>(operands[0].getDefiningOp());
        if (!getattr)
        {
            return mlir::failure();
        }
        arg_types.push_back(getattr.getOperand().getType());
        args.push_back(getattr.getOperand());
        llvm::copy(llvm::drop_begin(op.getOperandTypes(), 1), std::back_inserter(arg_types));
        llvm::copy(llvm::drop_begin(op.getOperands(), 1), std::back_inserter(args));
        name = extract_bound_func_name(name);
        // TODO kwargs
    }
    else
    {
        return mlir::failure();
    }

    return resolver(op, name, args, rewriter);
}
